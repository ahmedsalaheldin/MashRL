import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import json
from numpy import array
import zmq


class Task(object):

	def __init__(self,gWidth=5,gHeight=5):
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

		self.createGrid()

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

	def getView(self):
		return self.imgGrid[self.cursorX][self.cursorY] 

	def getStatus(self):
		if self.cursorX==self.GoalX and self.cursorY==self.GoalY:
			return "FINISHED" 
		else:
			return "OK"

	def reset(self):
		self.cursorX=0
		self.cursorY=0

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
		
	def Act(self,action):


		'''print "Xcur ",self.cursorX
		print "Xgoal ",self.GoalX
		print "Ycur ",self.cursorY
		print "Ygoal ",self.GoalY'''
		print action

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
			print "reached target ################################"			

		if self.cursorX<0:
			self.cursorX=0
			self.reward=-1
			print "X 0"
		elif self.cursorY<0:
			self.cursorY=0
			self.reward=-1
			print "Y 0"
		elif self.cursorX>self.gWidth-1:
			self.cursorX=self.gWidth-1
			self.reward=-1
			print "X max"
		elif self.cursorY>self.gHeight-1:
			self.cursorY=self.gHeight-1
			self.reward=-1
			print "X max"

		#print "X = ",self.cursorX
		#print "Y = ",self.cursorY
		
		
		

class Server(object):

	def __init__(self):
		context = zmq.Context()
		self.socket = context.socket(zmq.REP)
		self.socket.bind("tcp://*:6666")
		self.task = Task()



	def Receive(self):
		message = self.socket.recv()
		return message

	def Send(self,message):
		self.socket.send(message)


	def Handlemessage(self, message):


		if message=="GET_VIEW":	
			viewmessage = {'A':self.task.getView()}
			self.Send(json.dumps(viewmessage))#SEND THE VIEW

		elif message=="GET_ACTION":			
			self.Send(self.task.getAction())#SEND THE ACTION

		elif message=="GET_REWARD":	
			self.Send(str(self.task.getReward()))#SEND THE reward


		elif message=="RESET":	
	    		self.task.reset()
			self.Send("OK")#SEND OK


		elif message=="GET_STATUS":		
			self.Send(self.task.getStatus())

	
		else: 
			self.task.Act(message)
			#print "action = " , message
			self.Send("OK")



if __name__ == '__main__':


	server = Server()
	server.Receive()#receive task
	server.Send("OK");
	server.Receive()#receive environment
	server.Send("OK");

	

	while(1):		
		server.Handlemessage(server.Receive())
	
	

