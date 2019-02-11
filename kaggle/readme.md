The directory structure for the Kaggle project goes as follows:

data --|

  trainset --| Contains training images for cats and dogs. True images should be kept locally and not pushed to remote.
  
     Cat --|
     
     Dog--|
     
logs --| Store training logs. Useful for storing model parameters and training data.

models --| Store trained models.

notebooks --| Keep all notebooks here. All useful sections should be refactored into scripts. 

src --|
     
    data_processing --| Store scripts useful for generating and processing data
  
       dataset.py Keep dataset here
     
    models --|
  
      models.py : keep classification models here
    
    train_model.py : Script for training
  
    predict.py : Script for generating predictions for submission in correct format
  
utils --|
  
    model_utils.py : Keep functions useful for training and testing
  
    visualization.py : Keep functions for generating useful figures
  
