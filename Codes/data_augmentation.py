from keras.preprocessing import sequence, image
import random


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
                            
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=360,
                             
                             brightness_range=(1.0, 1.0),
                             zoom_range=[0.85,0.9],
                             fill_mode='nearest',
                             
                             )

i=0
for batch in datagen.flow_from_directory(
        'datsetpath',  # this is the target directory
        color_mode="rgb",
        target_size=(96, 96),
        batch_size=1,
        class_mode='binary',
        classes=['#'],
        save_to_dir='save_path' ,
        
        save_format='png',
        shuffle=False,
        
        ):
      
    i += 1
    
    
    if i > #int_number:
      break
        



