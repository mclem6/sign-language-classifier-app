Description:

This application is part of my ML research that can be found [here](https://github.com/mclem6/sign-language-classifier-research). This creates a website that
captures webcam feed and a pretrained model, which I applied transfer learning to, predicts which sign letter the user is signing. Because I trained the model
on MNIST dataset, which contained uniform 28 x 28 black and white images of a single person's hand signing all the letters, the model has a hard time with inference on a different domain from the one it was trained on. For this reason, model accuracy is very low on this application. Nonetheless, this is a great setup for anyone wanting to test their model, just replace the model in this file with your own!

Try it out [here](https://sign-language-classifier-app.vercel.app/)


Tech Stack
Python
FastAPI
MediaPipe






