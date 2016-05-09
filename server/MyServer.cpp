#include <MyServer.h>
#include <sstream>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>


using namespace std;
using namespace rapidjson;

MyServer::MyServer()
{
	//zmq::context_t _context (1);
	//zmq::socket_t _socket (_context, ZMQ_REP);
	//_socket.bind ("tcp://*:6666");

	//socket = &_socket;
	//context = &_context;

	_pApplicationServer= new SimulationServer(); 

	context = new zmq::context_t(1);
	socket = new zmq::socket_t(*context, ZMQ_REP); 	
	socket->bind ("tcp://*:6666");

	_taskParameters.clear();

	cout<<"done with constructor"<<endl;
}

MyServer::~MyServer()
{
}

string MyServer::Convert (float number)
{
    std::ostringstream buff;
    buff<<number;
    return buff.str();   
}

string MyServer::Receive()
{

	//zmq::socket_t _socket (*context, ZMQ_REP);
	//_socket.bind ("tcp://*:6666");

	zmq::message_t request;
        //  Wait for next request from client
	//cout<<"wait"<<endl;
        socket->recv (&request);
	//cout<<"done waiting"<<endl;

	string message_string = string(static_cast<char*>(request.data()), request.size());

        //std::cout << "Received Hello "<<message_string<< std::endl;

	return message_string;

}

void MyServer::Send(string message)
{
	zmq::message_t reply (message.length());
        memcpy (reply.data (), message.c_str(), message.length());
        socket->send (reply);
}

void MyServer::Handlemessage(string message)
{

	string  view, mimetype;
	size_t datasize;
	if(message=="GET_VIEW")
	{	
		//cout<<"getview"<<endl;
		unsigned char* pImage = _pApplicationServer->getView(view,datasize,mimetype);
		//Send(reinterpret_cast<const char*>(pImage));
	///////////////////////////////////////////////////////////
			float average;
			string msgstring;
			StringBuffer s;
			Writer<StringBuffer> writer(s);

			writer.StartObject();
			writer.String("A");
			writer.StartArray();

			for(int i=0;i<(datasize);i+=3)
			{
				average = 0.2126*(float)pImage[i] + 0.7152*(float)pImage[i+1] + 0.0722*(float)pImage[i+2];
				if(average<0){
					//average = 110000;
					cout<<average<<endl;}
				writer.Double((int)average);
			}
			writer.EndArray();
			writer.EndObject();

			msgstring= s.GetString();
	
			// send frame to prediction server
			Send(msgstring);
			delete[] pImage;
	/////////////////////////////////////////////////////////////

	}
	else if(message=="GET_ACTION")
	{	
		//cout<<"getaction"<<endl;
		Send(_pApplicationServer->getSuggestedAction());
	}
	else if(message=="GET_REWARD")
	{	
		//cout<<reward<<endl;
		Send(Convert(reward));

	}
	else if(message=="RESET")
	{	
    		_pApplicationServer->resetTask();
		Send("OK");

	}
	else if(message=="GET_STATUS")
	{	
		if(bFailed)
		{
			Send("FAILED");
		}
		else if(bFinished)
		{
			Send("FINISHED");
		}
		else
		{
			Send("OK");
		}

	}
	else 
	{	
		//cout<<message<<endl;
		_pApplicationServer->performAction(message, reward, bFinished, bFailed, strEvent);
		//cout<<"performed action"<<endl;
		Send(_pApplicationServer->getSuggestedAction());
		//cout<<"gotsuggestion"<<endl;
	}


}

void MyServer::Loop()
{

 	_pApplicationServer->setGlobalSeed(0);

	string task, environment;

	task = Receive();//receive task
	Send("OK");
	environment = Receive();//receive environment
	Send("OK");

	_pApplicationServer->initializeTask(task,environment);
	

	cout<<"after init"<<endl;	
	
	
	
	//for(int i=0;i<1;i++)
	while(1)
	{	
		
		Handlemessage(Receive());



		/*Receive();
		unsigned char* pImage = _pApplicationServer->getView(view,datasize,mimetype);
		Send("OK");
		task = Receive();
		_pApplicationServer->performAction(task, reward, bFinished, bFailed, strEvent);
		Send("OK");*/		
	}
}
