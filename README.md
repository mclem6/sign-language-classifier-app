Description:

This application is part of my ML research that can be found [here](https://github.com/mclem6/sign-language-classifier-research). This creates a website that
captures webcam feed and predicts which sign letter the user is signing using a pretrained model that I applied transfer learning to. Because I trained the model
on MNIST dataset, which contained uniform 28 x 28 black and white images of a single person hand singing all the letters, model has a hard time predicting webcam feed which is very
different from the domain it was trained on. For this reason, model accuracy is very low on this application. Nonetheless this is a great setup for anyone
wanting to test their model, just replace model on this file with your own!

Try it out [here](https://sign-language-classifier-app.vercel.app/)


Tech Stack
Python
FastAPI
MediaPipe






