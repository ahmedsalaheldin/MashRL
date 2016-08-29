import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import json
from numpy import array
import zmq
import numpy as np
import csv
import cv2


def resize_image(image,W,H):

	return cv2.resize(image,(W, H),interpolation=cv2.INTER_LINEAR)


class Task(object):

	def __init__(self,gWidth=30,gHeight=30):
		self.gWidth=gWidth
		self.gHeight=gHeight
		self.imgW=120
		self.imgH=90
		self.reward=0

		self.cursorX=0
		self.cursorY=0
		self.GoalX=self.gWidth-2
		self.GoalY=self.gHeight-1
		self.imgGrid= [None]*self.gWidth
		self.done = False

		self.createGrid()
		self.CreateheatMap()

	def createImage(self,text):

		font = ImageFont.truetype("OpenSans-Bold.ttf",50)
		img=Image.new('L', (self.imgW,self.imgH),(255))
		draw = ImageDraw.Draw(img)
		draw.text((0, 10),text,font=font)
		draw = ImageDraw.Draw(img)

		return array(img).tolist()

	def createGrid(self):


		#create an empty grid
		for i in range(self.gWidth):
			self.imgGrid[i] =[None]*self.gHeight

		#fill the grid
		for i in range(self.gWidth):
			for j in range(self.gHeight):
				number = (j+1)+(i*self.gHeight)
				self.imgGrid[i][j]=self.createImage(str(number))
				print "shape of image ", np.shape(self.imgGrid[i][j])

	def writeDataset(self,view,action):
		#view=view.reshape(1,-1)
		view=np.float32(np.asarray(view))
		print type(view)
		print type(view)
		view=resize_image(view,84,84)
		view=view.astype(np.uint8)
		img = Image.fromarray(view)
		img.save('mm.png')

		view=np.ravel(view)
		#print "shape = ", np.shape(view)


		if action=="TURN_LEFT":
			label ="0"
		elif action=="TURN_RIGHT":
			label ="1"
		elif action=="GO_FORWARD":
			label ="2"
		elif action=="GO_BACKWARD":
			label ="3"
		else:
		    raise ValueError('Unrecognized action.')

		with open('30images84.csv', 'a') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			writer.writerow(view)
		with open('30actions84.csv', 'a') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			writer.writerow(label)

	def getView(self):
		return self.imgGrid[self.cursorX][self.cursorY]

	def getFixedView(self,number):		
		return self.createImage(str(number))

	def getStatus(self):
		#print "cursorX = ",self.cursorX , "cursorY = ",self.cursorY
		if self.cursorX==self.GoalX and self.cursorY==self.GoalY:
			return "FINISHED" 
		else:
			return "OK"

	def reset(self):
		self.cursorX=0
		self.cursorY=0
		self.UpdateheatMap()

	def getReward(self):
		return self.reward


	def getAction(self):
		'''print "GXcur ",self.cursorX
		print "GXgoal ",self.GoalX
		print "GYcur ",self.cursorY
		print "GYgoal ",self.GoalY'''

		if self.cursorX>self.GoalX:
			return "TURN_LEFT"
		elif self.cursorX<self.GoalX:
			return "TURN_RIGHT"
		elif self.cursorY>self.GoalY:
			return "GO_BACKWARD"
		elif self.cursorY<self.GoalY:
			return "GO_FORWARD"

	def get_ALTAction(self):

		if self.cursorX<self.GoalX and  self.cursorY<self.GoalY:
			if self.cursorX>self.cursorY:
				return "GO_FORWARD"
			else :
				return "TURN_RIGHT"

		elif self.cursorX>self.GoalX:
			return "TURN_LEFT"
		elif self.cursorX<self.GoalX:
			return "TURN_RIGHT"
		elif self.cursorY>self.GoalY:
			return "GO_BACKWARD"
		elif self.cursorY<self.GoalY:
			return "GO_FORWARD"
		
	def Act(self,action):


		'''print "Xcur ",self.cursorX
		print "Xgoal ",self.GoalX
		print "Ycur ",self.cursorY
		print "Ygoal ",self.GoalY'''
		#print action
		#self.UpdateheatMap()
		self.reward=0
		if action=="TURN_LEFT":
			self.cursorX-=1
		elif action=="TURN_RIGHT":
			self.cursorX+=1
		elif action=="GO_FORWARD":
			self.cursorY+=1
		elif action=="GO_BACKWARD":
			self.cursorY-=1
		else:
		    raise ValueError('Unrecognized action.')

		

		if self.getStatus()=="FINISHED":
			self.reward=1
			#print "reached target ################################"			

		if self.cursorX<0:
			self.cursorX=0
			self.reward=-1
			#print "X 0"
		elif self.cursorY<0:
			self.cursorY=0
			self.reward=-1
			#print "Y 0"
		elif self.cursorX>self.gWidth-1:
			self.cursorX=self.gWidth-1
			self.reward=-1
			#print "X max"
		elif self.cursorY>self.gHeight-1:
			self.cursorY=self.gHeight-1
			self.reward=-1
			#print "Y max"

		#if self.cursorX==0 and self.cursorY==0:
		#	print "BAAAACKKKK TOO ZEEROOO"

		self.UpdateheatMap()

		#print "X = ",self.cursorX
		#print "Y = ",self.cursorY

	def CreateheatMap(self):
		self.heatmap_Train = np.zeros((self.gWidth,self.gHeight),dtype=np.int)
		self.heatmap_Test = np.zeros((self.gWidth,self.gHeight),dtype=np.int)
		self.UpdateheatMap()
		print self.heatmap_Train


	def UpdateheatMap(self):
		if self.done:
			self.heatmap_Test[self.cursorX,self.cursorY]+=1
		else:
			self.heatmap_Train[self.cursorX,self.cursorY]+=1
		
		
def Play():

	task = Task()
	num_episodes=5
	for i in range(num_episodes):	
		status ="start"
		print "new episode"
		task.reset()
		while(status!="FINISHED"):
			view=task.getView()
			action =task.get_ALTAction()
			task.writeDataset(view,action)
			task.Act(action)
			status = task.getStatus()
			

			

class Server(object):

	def __init__(self):
		context = zmq.Context()
		self.socket = context.socket(zmq.REP)
		self.socket.bind("tcp://*:6666")
		self.task = Task()
		self.counter=1



	def Receive(self):
		message = self.socket.recv()
		return message

	def Send(self,message):
		self.socket.send(message)


	def Handlemessage(self, message):


		if message=="GET_VIEW":	
			viewmessage = {'A':self.task.getView()}
			#viewmessage = {'A':self.task.getFixedView(self.counter%900)}
			self.Send(json.dumps(viewmessage))#SEND THE VIEW
			self.counter+=30

		elif message=="GET_ACTION":			
			#self.Send(self.task.getAction())#SEND THE ACTION
			self.Send(self.task.get_ALTAction())

		elif message=="GET_REWARD":	
			self.Send(str(self.task.getReward()))#SEND THE reward


		elif message=="RESET":	
	    		self.task.reset()
			self.Send("OK")#SEND OK

		elif message=="DONE":	
	    		self.task.done = not (self.task.done)
			self.Send("OK")#SEND OK
			if self.task.done:
				print self.task.heatmap_Train
			else:
				print self.task.heatmap_Test

		elif message=="GET_STATUS":		
			self.Send(self.task.getStatus())

	
		else: 
			self.task.Act(message)
			#print "action = " , message
			self.Send("OK")

		
def Serve():

	server = Server()
	server.Receive()#receive task
	server.Send("OK");
	server.Receive()#receive environment
	server.Send("OK");

	

	while(1):		
		server.Handlemessage(server.Receive())


if __name__ == '__main__':


	Serve()
	#Play()
	

