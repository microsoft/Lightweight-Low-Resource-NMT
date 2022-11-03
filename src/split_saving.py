from transformers import TFMT5ForConditionalGeneration, T5Tokenizer
import sentencepiece as spm
import tensorflow as tf 
import os 
import numpy as np
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def getMT5CustomEncoder(encoder, config):
    input_ids =  tf.keras.layers.Input(shape=(encoder_max_len, ), dtype = tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(encoder_max_len, ), dtype = tf.int32) 
    encoded_sequence = encoder(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = config.output_hidden_states)
    return tf.keras.Model(inputs = [input_ids, attention_mask], outputs = [encoded_sequence])

def getMT5CustomDecoder(layer, config):
    decoder = layer.decoder
    lm_head = layer.lm_head
    input_ids =  tf.keras.layers.Input(shape=(encoder_max_len, ), dtype = tf.int32)
    decoder_input_ids = tf.keras.layers.Input(shape=(decoder_max_len, ), dtype = tf.int32) 
    decoder_attention_mask = tf.keras.layers.Input(shape=(decoder_max_len, ), dtype = tf.int32) 
    encoder_outputs =  tf.keras.layers.Input(shape=(encoder_max_len, config.d_model), dtype = tf.float32) 
    decoder_outputs = decoder(tf.convert_to_tensor(decoder_input_ids),encoder_hidden_states=encoder_outputs)
    lm_logits = lm_head(decoder_outputs[0])
    return tf.keras.Model(inputs = [decoder_input_ids, encoder_outputs], outputs = [lm_logits])

def getTFBCustomEncoder(encoder, config):
   
    input_ids =  tf.keras.layers.Input(shape=(ENCODER_MAX_LEN,), dtype = tf.int32) 
    attention_mask = tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32)
    encoded_sequence = encoder(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = config.output_hidden_states)
    return tf.keras.Model(inputs = [input_ids, attention_mask], outputs = [encoded_sequence])

def getTFBCustomDecoder(layer, config):
    decoder = layer.decoder
    shared = layer.shared
    input_ids =  tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) 
    decoder_input_ids = tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) 
    decoder_attention_mask = tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, ), dtype = tf.int32) 
    encoder_outputs =  tf.keras.layers.Input(shape=(ENCODER_MAX_LEN, config.d_model), dtype = tf.float32) 
    
    decoder_outputs = decoder(tf.convert_to_tensor(decoder_input_ids),encoder_hidden_states=encoder_outputs)    
    first_dims = shape_list(decoder_outputs[0])[:-1]
    x = tf.reshape(decoder_outputs[0], [-1, config.d_model])
    logits = tf.matmul(x, shared.weight.numpy(), transpose_b=True)
    lm_logits = tf.reshape(logits, first_dims + [config.vocab_size])
  
    return tf.keras.Model(inputs = [decoder_input_ids, encoder_outputs], outputs = [lm_logits])

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--model_arch", type=str, default = "marian")  
    parser.add_argument("--model_path", type=str)    
    parser.add_argument("--max_encoder_length", type=str)
    parser.add_argument("--max_decoder_length", type=str)

    encoder_max_len = args.max_encoder_length
    decoder_max_len = args.max_decoder_length 
    
    encoder_tflite_path = args.model_path + 'encoder.tflite'
    decoder_tflite_path = args.model_path + 'decoder.tflite'

    args = parser.parse_args()

    if args.model_arch == 'mt5':

        model = TFMT5ForConditionalGeneration.from_pretrained(args.model_path, from_pt = True)
        model.save_pretrained(args.model_path)
        model = TFMT5ForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    
        encoder = getMT5CustomEncoder(encoder = model.get_encoder(), config = model.config)
        converter = tf.lite.TFLiteConverter.from_keras_model(encoder)
        print('Optimizing Encoder Size Reduction with the Keras Model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
        
        encoder_tflite_model = converter.convert()    
    
        decoder = getMT5CustomDecoder( layer = model, config = model.config)

        converter = tf.lite.TFLiteConverter.from_keras_model(decoder)
        print('Optimizing Decoder Size Reduction with the Keras Model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]

        decoder_tflite_model = converter.convert()    

        
    if args.model_arch == 'marian': 
        model = TFMarianMTModel.from_pretrained(args.model_path)
        encoder = getTFBCustomEncoder(encoder = model.model.encoder, config = model.model.config)
        print('Converting Encoder to TFlite')
        converter = tf.lite.TFLiteConverter.from_keras_model(encoder)
        print('Optimizing Encoder Size Reduction with the Keras Model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
        converter.experimental_enable_resource_variables = True
        encoder_tflite_model = converter.convert()    
        
        decoder = getTFBCustomDecoder(layer = model.model, config = model.model.config)
        print('Converting Decoder to TFlite')
        converter = tf.lite.TFLiteConverter.from_keras_model(decoder)
        print('Optimizing Decoder Size Reduction with the Keras Model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
        converter.experimental_enable_resource_variables = True
        decodertflite_model = converter.convert()    
    
    
    print('Writing Encoder and Decoder TFLite Files')
    with open(encoder_tflite_path,'wb') as file:
            file.write(encoder_tflite_model)

    with open(decoder_tflite_path,'wb') as file:
        file.write(decoder_tflite_model)
