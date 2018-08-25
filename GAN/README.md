# GENERATIVE ADVERSAL NETWORKS

Generative adversarial networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework. They were introduced by Ian Goodfellow et al. in 2014.This technique can generate photographs that look at least superficially authentic to human observers, having many realistic characteristics (though in tests people can tell real from generated in many cases).

The Two Networks: 
    Generator:
    Discriminator
    
    
The Generators generates the fake candidate (picture or a speech signal) and sends it to the discriminator for the judgement which is being trained by the real candidates simultaneously. The discriminator is then trained to discriminate the fake picture to be true or not.Then the error is backpropageted to generator and it inturn makes some changes and sends the candidate again till the discriminator is fooled :P

## Real Picture
![real_samples](https://user-images.githubusercontent.com/37619070/44615320-0ab32800-a855-11e8-83da-13c07b096972.png)

## Fake picture; epoch == 0
![fake_samples_epoch_000](https://user-images.githubusercontent.com/37619070/44615493-d5a8d480-a858-11e8-8830-417b53e39e0e.png)

## Fake picture; epoch == 12
![fake_samples_epoch_012](https://user-images.githubusercontent.com/37619070/44615498-f40ed000-a858-11e8-9ab7-4feddb5837b3.png)

## Fake picture; epoch == 24
![fake_samples_epoch_024](https://user-images.githubusercontent.com/37619070/44615503-0b4dbd80-a859-11e8-9935-a0d88721cd4e.png)


For more intuitive understanding:
  
  https://medium.freecodecamp.org/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394
  
  https://medium.com/@Moscow25/gans-will-change-the-world-7ed6ae8515ca
