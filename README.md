**WORKING OF AN LSTM CELL**

LSTMs work in a three-step process.

•The first step in LSTM is to decide which information to be omitted from 
the cell in that particular time step. It is decided with the help of a sigmoid 
function. It looks at the previous state (ht-1) and the current input xt and 
computes the function.

•There are two functions in the second layer. The first is the sigmoid 
function, and the second is the tanh function. The sigmoid function decides 
which values to let through (0 or 1). The tanh function gives the weightage 
to the values passed, deciding their level of importance from -1 to 1.

•The third step is to decide what will be the final output. First, you need to 
run a sigmoid layer which determines what parts of the cell state make it to 
the output. Then, you must put the cell state through the tanh function to 
push the values between -1 and 1 and multiply it by the output of the 
sigmoid gate

![image](https://github.com/user-attachments/assets/2a50ed0c-280e-40b9-82ea-a7bdb4d0e3c0)


**RESULTS AND OUPUT**

![image](https://github.com/user-attachments/assets/ebad3bbb-fae8-40da-88a7-6f77f0b73a02)
![image](https://github.com/user-attachments/assets/a68e5fde-9852-4335-8b20-ce7949dfeead)
![image](https://github.com/user-attachments/assets/e31777ea-b321-4d57-b2d6-3416a977d0d6)


