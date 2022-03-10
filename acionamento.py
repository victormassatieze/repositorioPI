import RPi.GPIO as GPIO
import time
import os

#adjust for where your switch is connected
buttonPin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(buttonPin,GPIO.IN)

os.system('aplay test_botao.wav')
#initialise a previous input variable to 0 (assume button not pressed last)
prev_input = 0
while True:
  #take a reading
  input = GPIO.input(buttonPin)
  #if the last reading was low and this one high, print
  if (input == 0):
    os.system("python3 /home/pi/projetoIntegrado/0grava.py")
  #update previous input
  input = GPIO.input(buttonPin)
  #slight pause to debounce
  time.sleep(0.05)

      