from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout

total_signs = 25

def get_model():
  myModel = Sequential();
  myModel.add( Conv2D( 128, (2,2), activation='relu', input_shape=(28,28,1) ) );
  myModel.add( Conv2D( 64, (2,2), activation='relu' ) );
  myModel.add( Dropout(0.3) );
  myModel.add( Conv2D( 64, (2,2), activation='relu' ) );
  myModel.add( Flatten() )
  myModel.add( Dense( 512, activation='relu' ) );
  myModel.add( Dropout(0.5) );
  myModel.add( Dense( 128, activation='relu' ) );
  myModel.add( Dense( total_signs, activation='softmax' ) );
  myModel.compile( optimizer='adaDelta', loss='categorical_crossentropy', metrics=['accuracy']);
  myModel.summary()
  return myModel;