# Climate-Sensing

Climate Change, a highly debated topic across the globe, but do we really care about it? We see politicians, scientists, researchers, philanthropists, conservationists and almost everyone blaming each other for every climate catastrophe occurring today. The real question is what climate change or change are we talking about?



Objective and Intent of  project is Using Machine Learning  to detect pests that will harm Agriculture and Farming, cutting trees using audio analysis. Also predicting fire, greenhouse adaptation and plant growth.

Solution Approach 
 how to build a device using TensorFlow and Artemis module to solve most of the problems faced by the rural or agricultural communities anywhere.The device uses Machine Learning algorithms to check overall plant health, extreme climate prediction and protection, auto-greenhouse adaptation and detecting deadly disease spreading vectors or illegal logging of forests using audio analysis.The project also demonstrates how we can have more insights for our farm just by collecting and utilizing the data got from our sensors. I got inspiration for making this project from news like Australian wildfires, Indian GDP falling down due to faulty agricultural practices and locust swarm which is damaging the crops in East Africa and many other nations veryrapidly, mosquitoes breeding at an alarmingrate,neglected tropicaldiseases, so I found a call within me as an active member of this innovative, committed community.

The important measure taken while making this project was collecting data efficiently and patiently, once the data is collected the job becomes lot more easier  or else  are bound to fail if you don't collect correct data to feed your hungry ML frameworks


Salient Features Included
1)Farm Analysis:

With increasing population a sustainable method of farming is important. Previously we just used to play with sensors but now using Tensorflow we can not only sense but analyse, predict and take actions. Using all the collected data we will find abnormalities in crop growth, photosynthesis rate, extreme climate and need for smart green house adaptation. 

2) Detection of certain beneficial andharmful organisms using audio analysis :

3)Stopping deforestation :

4)Predicting wildfires and adapting for extreme situations:


Audio Classification:
Step 1: Collecting the required dataset for audio analysis:

We got audio files from dataset for different kinds of mosquito wing beat, honeybee, you can too download it from there, the file is big but we don't need that much dataset since our device memory is small, so we will limit our self to 200 audio files of each class only, of one seconds utterance each. Similarly you can get for chainsaw, locust, cricket in this way. You can also record your own data if you have a silent room with insects, that would be much better and accurate and that would be very much exciting but I wasn't able to do so because of my stressful school exams. Note: If we directly load the audio data downloaded for the training process it won't work because the microphone architecture would differ and the device won't recognize anything. So once the files are downloaded we need to record it again via the Artemis microphone so that we can train on the data which is exact to be used for running inference. So let's configure our Arduino IDE for Artemis Redboard ATP, look the below link for that.

https://learn.sparkfun.com/tutorials/hookup-guide-for-the-sparkfun-redboard-artemis-atp

Select the board as Artemis ATP then from File->Examples->Sparkfun Redboard Artemis Example->PDM->Record_to_wav

Along with the code there comes a python script which you need to run to record audio from the on-board mic. It is very necessary because audio files came from different mics so there might be a chance that the board would not be able to recognize the frequencies accurately and treat the sound as noise


I used Audacity to clean up my downloaded audio files before I slice them up, also there are very awesome features which you can use to know whether you audio is pure or not, audio spectrogram.


Once the audio files are sliced up you are ready to train those files for your Artemis. However you need to tweak to make it work perfectly because the data might contain lots of noise due to enclosure or working environment hence I recommend you to train background dataset also so that it can work even there is constant peculiar noise. Background contains all the trimmed audio segment in Audacity software and some distinct noise.

For training I used Google Colab, below is full training process images, you need to upload these files while training in Colab notebook and run on GPU, for me it took nearly 2 hours. I had to train thrice because the notebook kept disconnecting due to my slow internet connection, so I trained first a small dataset only bee, mosquito(no gene or breed classification) and chainsaw after that I trained the whole dataset for two times and luckily I succeeded.

Step 2:Training theaudio data

Audio Training #1: Labels = mosquito, bee, chainsaw

Once the training is complete we need to freeze the model and convert it into lite model to be used in Edge device. I will attach the code along with all required comments.

Audio Training #2: Labels = aedes, culex, anopheles, bee, chainsaw

Download all the files generated from the above steps, also you need to generate microfeature file for every class audio file, I will explain this when we code the device. Also you can read the TensorFlow Lite for Microcontrollers documentation for more information.

Data Classification:
Step 1: Collecting sensor datafor both desired and undesired behaviors

1) Plant height: I used my past project to find out rice plant height after germination but that too was not sufficient because I could keep my device on for so long as there is no SigFox connectivity in India so whatever reading I got I used Excel mathematical statistical functions, regression technique to find all other points and added some noise. I am really sorry for having small dataset for ML but thankfully it suits my application, if I someday make it a business product then I would expand my dataset by generating synthetic data.

2)  Fire prediction:  We used the environmental combo quiic sensor to collect the normal readings as well as fire situation reading. For fire we need himidiy, temperatures, tVOC, CO2 readings. So basically we are using making a classification model using regression technique to make predictions. I used the example code of the sensor to take all the readings. It is very essential feature to detect wildfires.

3)Greenhouse Adaptation: I used quiic VCNL4040 module to detect ambient light levels and environmental combo sensor for temperature and CO2 sensing. Based on the data the device can predict when to adapt to greenhouse mode to save the crops. It can be further optimized to protect the crop from hailstorms or heavy snowfalls.

Step 2:Training the data using deep learning

Never proceed any training without normalization as you will never see your model improving it's accuracy. Below are some images for my training process, it is good enough for anyone to know what is happening during the training.

Training #1: Fire prediction

I used 'adam' optimizer and 'binary_crossentropy' loss, they worked better than any other methods, you can use whatever accordingly. The same training step was followed for other datasets also. Also sigmoid is used as activation functiuon for output layer as sigmoid is good for non linear data and binary predictions

Training #2: Greenhouse adaptation prediction

Same regression model configuration for neural network was used as that of the fire prediction model. The output prediction will help the user to know on the basis of relevant sensor data whether the plants are getting enough CO2, light, warmth or not if not then the device can take action to adapt to greenhouse( I will tell how when we will code the device).

Training #3: Plant growth tracking

Here instead of adam optimizer rmsprop was used and mse as loss function. In this I have trained the underlying neural network to identify the pattern of growth with regard to days, so I will use it to compute plant height from model based upon days and check with my sensor readings if the values differ greatly it means the plant is not growing well in that season.

Since we have completed all our training steps we will head toward programming the device.

Test #1: Audio Analysis-

Checking our speech model and displaying results on the OLED screen. If you are not confident about programming the whole project you can pick up example project and keeping it as a base modify it, you will easily learn to debug also in this manner. The images don't show in detail hence I have uploaded the codes but you will still need to tweak it as per your project. You can also easily modify it to send the results to a mobile or base stations on the basis of detected sounds, like mosquito, so one can easily see where are the most occurrence of such insects in the locality and prevent it's breeding or make automated drone navigation system to spray particular insecticides, pesticides in farm areas or inform agencies about illegal logging with full precision control. Look below for our test result images.

We have completed our training and programming the audio analysis model for our project, now I will proceed with programming other value based output models.

Test #2: Fire prediction-

We will load the data model on the board and then perform real-time predictions for our sensor values, based upon the prediction we will will display the notification on OLED screen, thereby eliminating the chance of widespread of fire in farms. This really took most of my time since my board was giving me compilation error every time however hard I tried. But thanks to Hackster community, who helped me to upload the code successfully.

Test #3: Greenhouse prediction

Here we will utilize ambient light, temperature and CO2 concentration detection sensor to predict whether there is sufficient light, temperature and CO2 available for plant growth or not, if the condition is not normal we will direct servo motors activate and protect the plants by bringing a layer over the farm like Greenhouse set-up.

Test #4: Plant Height Determination

We have trained our model to determine plant height through regression but with regression there is one drawback, the drawback is that it determines the values in one linear direction only therefore if left to run forever it would return height of plant in kilometers some day if no upper limit is set. So what we will do is that we will detect height of the plant for corresponding days and check with our model for corresponding day and check whether the height predicted by model differs widely from our measured plant height, it differs widely it means there is some nutritional deficiency.

I have attached all the codes which are commented very well and readable enough for you to implement your own algorithm, don't waste your time thinking about codes, invest your time to make your training dataset as good as possible.

I don't have a 3D printer so I just took a plastic case to build my model, during idea submission also we were required to specify how would our project look like, so here is the installation(not garden as I don't have one) but I tried my level best to do as much innovation and creativity possible at my end. Sorry guys, I couldn't find fritzing parts for this hardware but wiring is fairly simple as I used very common sensors and quiic sensors are plug and play, all images are crisp enough to detail out pins for my sensors.

You can just set your device to whatever mode you want and wherever you want. I also wanted to implement moisture adaptation for my plants I will finish that too later.

SDG 3: GOOD HEALTH AND WELL-BEING

Using our device we have tried to make locality aware of all the disease carrying vectors present or breeding also we are able to save the crop as soon as we hear sound made by any pest, so it saves lots of efforts of farmer and makes them spray chemicals only when required, eat healthy.

