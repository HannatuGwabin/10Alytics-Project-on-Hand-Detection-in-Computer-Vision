#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mediapipe')


# In[2]:


# step 1: Import all necessary libraries
import cv2
import mediapipe as mp


# In[3]:


# step 2: identify webcam
cap = cv2.VideoCapture(0) # local cam 0 and external cam 1


# In[4]:


# leveraging the mediapipe library used for hand detection

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# In[5]:


# step 3: switch on your webcam

while True:
    _, img = cap.read()
     
    # convert image from BG to RGB
    imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply mediapipe
    results = hands.process(imgRGB)
    
    print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    
    cv2.putText(img, "10Alytics Hand Detection Program", (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
    cv2.imshow("10ALYTICS Hand Detection Project", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release the capture once all the processing is done
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




