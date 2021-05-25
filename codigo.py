%tensorflow_version 2.x
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#ruta raiz
PATH = "/content/drive/MyDrive/ProyectoIA"
#entrada
INPATH = PATH + '/input'
#salida
OUTPATH = PATH + '/output'
#checkpoint
CHECKP = PATH + '/checkpoints'
#listado de archivo de inpath
imgurls = !ls -1 "{INPATH}"
n=228
train_n=round(n*0.80)
#Listado randomizado 
randurls=np.copy(imgurls)
#np.random.seed(23) #solo el tutorial
np.random.shuffle(randurls)
#Particiom train/tet
tr_urls=randurls[:train_n]
ts_urls= randurls[train_n:n]
print(len(imgurls),len(tr_urls),len(ts_urls))
#rescalar imagenes
IMG_WIDTH=256
IMG_HEIGHT=256

def resize(inimg,tgimg,height,width):
#funcion de rezise para reescalar las imagenes
  inimg=tf.image.resize(inimg,[height,width])
  tgimg=tf.image.resize(tgimg,[height,width])
  return inimg,tgimg

#normaliza el rango[-1, +1] en lugar de 0 a 255
def normalize(inimg,tgimg):
  inimg=(inimg/127.5)-1
  tgimg=(tgimg/127.5)-1
  return inimg,tgimg

@tf.function()
#aumentacion de datos :random crop+flip
#cumple la necesidad de aumentar datos para generar de una imagen mas imagenes para un mejor entrenamiento
def random_jitter(inimg,tgimg):
  inimg,tgimg = resize(inimg,tgimg,286,286)
#apilar las imagenes en el eje de una imagen a un objeto 
  stacked_image=tf.stack([inimg,tgimg], axis=0)
  cropped_image = tf.image.random_crop(stacked_image, size=[2,IMG_HEIGHT,IMG_WIDTH,3])
  
  inimg,tgimg=cropped_image[0],cropped_image[1]
  
  if tf.random.uniform(()) > 0.5 :
    inimg=tf.image.flip_left_right(inimg)
    tgimg=tf.image.flip_left_right(tgimg)
    
  return inimg,tgimg

def load_image(filename, augment=True):
#limitar la ultima funcion tenga 3 componentes
  inimg=tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH+'/'+filename)),tf.float32)[..., :3]
  tgimg=tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUTPATH+'/'+filename)),tf.float32)[..., :3]
  
  inimg,tgimg=resize(inimg,tgimg,IMG_HEIGHT,IMG_WIDTH)
  if augment:
    inimg,tgimg = random_jitter(inimg,tgimg)
  inimg,tgimg= normalize(inimg,tgimg)
  return inimg, tgimg   
  ##encapsulan loadimage    
def load_train_image(filename):
  return load_image(filename,True)

def load_test_image(filename):
  return load_image(filename,False)

plt.imshow(((load_train_image(randurls[0])[1]) + 1) / 2)

#-----------------------------
#generar un dataset a partir de las imagenes
#tomar las imagenes de google drive
train_dataset=tf.data.Dataset.from_tensor_slices(tr_urls)
#se encargara del mapeo de las imagenes
train_dataset=train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#distribuye en diferentes lotes
train_dataset=train_dataset.batch(1)

#nos devolvera 5 lotes de imagenes
for inimg,tgimg in  train_dataset.take(5):
    plt.imshow(((tgimg[0,...]) + 1) / 2)
    plt.show()
    
#ahora se hara generador con el test
test_dataset=tf.data.Dataset.from_tensor_slices(ts_urls)
test_dataset=test_dataset.map(load_test_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset=test_dataset.batch(1)

#----------------generador
from tensorflow.keras.layers import *
from tensorflow.keras import *

def downsample(filters,apply_batchnorm=True):
  #ruido aleatorio
  initializer=tf.random_normal_initializer(0,0.02)
  result=Sequential()
  #capa convolucional 
  result.add(Conv2D(filters, 
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer,
                    use_bias=not apply_batchnorm))
  if apply_batchnorm:
  #capa de batch normalization
      result.add(BatchNormalization())
  #capa de activacion
  result.add(LeakyReLU())
  return result
downsample(64)


def upsample(filters,apply_dropout=False):
  initializer=tf.random_normal_initializer(0,0.02)
  result=Sequential()
  #capa convolucional
  result.add(Conv2DTranspose(filters, 
                              kernel_size=4,
                              strides=2,
                              padding="same",
                              kernel_initializer=initializer,
                              use_bias=False))
  result.add(BatchNormalization())
  if apply_dropout:
  #añade capa de batch normalization
      result.add( Dropout(0.5))
  #capa de activacion
  result.add(ReLU())
  return result
upsample(64) 

#
def Generator():
#especificar la entrada corresponde a que la imagen tiene diferentes tamaños
#el 3 corresponde a los canales rgb
  inputs=tf.keras.layers.Input(shape=[None,None,3])

#conjunto de bloques que corresponden a la red
#los datos especificados corresponden a la arquitectura del ecoder-decoder
#encoder C64-C128-C256-C512-C512-C512-C512-C512
  down_stack=[
              downsample(64, apply_batchnorm=False),
              downsample(128),
              downsample(256),
              downsample(512),
              downsample(512),
              downsample(512),
              downsample(512),
              downsample(512),
  ]
#encoder C512-C512-C512-C512-C512-C256-C128-C64
  up_stack=[
            upsample(512,apply_dropout=True),
            upsample(512,apply_dropout=True),
            upsample(512,apply_dropout=True),
            upsample(512),
            upsample(256),
            upsample(128),
            upsample(64),
  ]

  initializer=tf.random_normal_initializer(0,0.02)
  last=Conv2DTranspose(filters=3,
                        kernel_size=4,
                        strides=2,
                        padding="same",
                        kernel_initializer=initializer,
                        activation='tanh')#tangente hipervolica por venir de -1 a 1
  x=inputs
  s= []  
  concat=Concatenate()
#iterar toma la capa down y pasa el resultado de la itercion anterior
#basicamente tomar el resultado de la imagen anterior y pasarla a la siguiente 
  for down in down_stack:
    x= down(x)
    s.append(x)
  
  s =reversed(s[:-1]) 
  
  for up ,sk in zip (up_stack,s):
    x=up(x)
    x=concat([x,sk])

  last=last(x)
  #regresa a un modelo generador
  return Model(inputs=inputs,outputs=last )
generator=Generator()
gen_output=generator(((inimg+1)*255), training=False)
plt.imshow(((inimg[0,...]) + 1) /2)
plt.show()
plt.imshow(gen_output[0,...])

#--------------
#discrimidador recibe dos entradas
def Discriminator():
  #imagen destino que clasifica como real
  #imagen origen que clasifica como falsa
  ini=Input(shape=[None,None,3],name='input_img')
  gen=Input(shape=[None,None,3],name='gener_img')
  con=concatenate([ini,gen])
  initializer=tf.random_normal_initializer(0,0.2)
  down1=downsample(64,apply_batchnorm=False)(con)
  down2=downsample(128)(down1)
  down3=downsample(256)(down2)
  down4=downsample(512)(down3)

  last=tf.keras.layers.Conv2D(filters=1,
                              kernel_size=4,
                              strides=1,
                              kernel_initializer=initializer,
                              padding="same")(down4)
  return tf.keras.Model(inputs=[ini,gen],outputs=last)

discriminator=Discriminator()       
disc_out=discriminator([((inimg+1)*255),gen_output], training=False)
plt.imshow(disc_out[0,...,-1],vmin=20,vmax=20,cmap='RdBu_r')
plt.colorbar()
disc_out.shape

#----------
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True)

#perdida del discriminador toma 2 entradas, imágenes reales e imágenes generadas
def discriminator_loss(disc_real_output, disc_generated_output):
  #real_loss es una pérdida de entropía cruzada sigmoidea de las imágenes reales y una matriz de unas
  real_loss=loss_object(tf.ones_like(disc_real_output),disc_real_output)
  #generate_loss es una pérdida de entropía cruzada sigmoidea de las imágenes generadas y una matriz de ceros
  generated_loss=loss_object(tf.zeros_like(disc_generated_output),disc_generated_output)

#Entonces la pérdida_total es la suma de la pérdida_real y la pérdida_generada
  total_disc_loss=real_loss + generated_loss

  return total_disc_loss
#perdida de entropia de la serie de imagenes generadas
LAMBDA=100

def generator_loss(disc_generated_output,gen_output,target):
  gan_loss=loss_object(tf.ones_like(disc_generated_output),disc_generated_output)

  #mean absolute error
  #perdida del error absoluto entre la imagen generada y la objetivo
  #permitiendo que la imagen generada sea similar a la objetivo
  l1_loss=tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss= gan_loss + (LAMBDA* l1_loss)
  return total_gen_loss

#Escribe una función para trazar algunas imágenes durante el entrenamiento.
#Pasamos imágenes del conjunto de datos de prueba al generador.
#El generador luego traducirá la imagen de entrada en la salida.
#El último paso es trazar las predicciones
def generate_images(model,test_input, tar, save_filename=False,display_imgs=True):
  prediction=model(test_input,training=True)
  if save_filename:
    tf.keras.preprocessing.image.save_img(PATH + '/outputB/'+save_filename +'.jpg', prediction[0,...])
  plt.figure(figsize=(10,10))

  display_list=[test_input[0],tar[0],prediction[0]]
  title=['Input image','Ground Truth','Predicted Image']
  if display_imgs:
    for i in range(3):
      plt.subplot(1,3,i+1)
      plt.title(title[i])
      plt.imshow(display_list[i]*0.5+0.5)
      plt.axis('off')
  plt.show()

@tf.function() 
#cada que se llame la funcin se le manda la imagen de entrda y la imagen resultante

def train_step(input_image,target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
    #obtiene una imagen de salida y entrene en el paso de comprimir descomprimir la imagen
    output_image =generator(input_image,training=True)
    #observara la imagen que obtiene y evaluaran si es real o no
    output_gen_discr=discriminator([output_image,input_image],training=True)
    output_trg_discr=discriminator([output_image,input_image],training=True)
    discr_loss=discriminator_loss(output_trg_discr,output_gen_discr)
    gen_loss=generator_loss(output_gen_discr,output_image,target)

    generator_grads=gen_tape.gradient(gen_loss,generator.trainable_variables)
    discriminator_grads=discr_tape.gradient(discr_loss,discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_grads,generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grads,discriminator.trainable_variables))


#El ciclo de entrenamiento itera sobre el número de épocas
from  IPython.display import clear_output

def train(dataset, epochs):
#En cada época borra la pantalla y ejecuta generate_images para mostrar su progreso.
#En cada época, itera sobre el conjunto de datos de entrenamiento
  for epoch in range (epochs):
    imgi=0
    for input_image, target in dataset:
      print('epoch'+str(epoch)+'-train:' +str(imgi)+'/'+str(len(tr_urls)))
      imgi+=1
      train_step(input_image,target)
      clear_output(wait=True)
    imgi=0
    for inp,tar in test_dataset.take(5):
      generate_images(generator,inp,tar,str(imgi)+'_'+str(epoch),display_imgs=True)
      imgi+=1

train(train_dataset,200)
