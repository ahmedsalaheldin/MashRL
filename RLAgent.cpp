#include <stdio.h>
#include <iostream>
#include <fstream>
#include <zmq.hpp>
#include <zmq_utils.h>
#include <string>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <mash-network/client.h>
//#include <Image.h>
//#include <svm.h>
#include <sstream>

using namespace std;
using namespace Mash;
using namespace rapidjson;


string Convert (float number)
{
    std::ostringstream buff;
    buff<<number;
    return buff.str();   
}


int main(int argc, char** argv)
{
	const int width = 120; //width of frame
	const int height = 90; //height of frame
	const int datasize = width*height*3; // 3 channel image size
 	int counter=0 , timeOut = 5000 , numFinish=0; //time counter perlevel  time out =500 for flag, much larger for line 
	ofstream result("score.csv");
	ifstream features;
	OutStream* out = new OutStream(0);
	Client* c = new Client(out);
	unsigned char data[datasize];
	ArgumentsList arg ;
	ArgumentsList actions ;
	ArgumentsList empty;
	//string task = "reach_1_flag"; // reach flag task
	string task = "follow_the_line"; // follow line task
	//string environment = "SingleRoom"; // reach flag task
	string environment = "Line"; // follow line task
	int frame_skip = 1;
	bool predictFlag;
	bool testing=0;
	bool roundended;


///////////Connect to MASH Server/////////////
	zmq::context_t context (1);
	zmq::socket_t socket (context, ZMQ_REQ);
	zmq::message_t reply;

	socket.connect ("tcp://localhost:6666"); // if client
	cout<<"connected"<<endl;
////////////////////////////////////////////////////////
///////////Connect to Prediction Server/////////////
	zmq::context_t context2 (2);
	zmq::socket_t socket2 (context2, ZMQ_REQ);


	socket2.connect ("tcp://localhost:5555"); // if client
	cout<<"connected"<<endl;

	// declare zmq message variables 

	StringBuffer s;
	Writer<StringBuffer> writer(s);

	string msgstring;
	//int msgsize;

	//zmq::message_t reply;

	string predictionStr;

	string reward_string;

	StringBuffer rsb;
	Writer<StringBuffer> rwriter(rsb);

	string rmessage;

////////////////////////////////////////////////////////

	
	string replystr;
	string replyaction;
	string replyframe;

	// Send Task
	cout<<"sending"<<endl;
	zmq::message_t msg_task (task.length());
	memcpy ((void *) msg_task.data (), task.c_str(), task.length());
	socket.send (msg_task);	

	//receive ok
	cout<<"receiving"<<endl;
	socket.recv (&reply); // 
	replystr = string(static_cast<char*>(reply.data()), reply.size());
	cout<< replystr <<endl;

	// Send environment
	zmq::message_t msg_env (environment.length());
	memcpy ((void *) msg_env.data (), environment.c_str(), environment.length());
	socket.send (msg_env);	

	//receive ok

	socket.recv (&reply); // 
	replystr = string(static_cast<char*>(reply.data()), reply.size());
	cout<< replystr <<endl;
	


	/////////////////////////////////
	/////START PLAYING///////////////
	/////////////////////////////////

	cout<<"im gonna start playing"<<endl;

	arg.clear();
	arg.add("main");
	for(int i=0; i<1000000;i++) //number of rounds
	{	

		predictFlag=1;	
		counter=0; // number of frames elapsed in a round
		roundended=0;
		//counter < timeOut &&
		while(replystr != "FINISHED" &&  replystr!= "FAILED") // while the round hasn't failed or succeeded yet, and time hasn't run out
		{

		    // Declare non terminal DQN
		    msgstring = "NOTTERMINAL";
		    zmq::message_t terminal (msgstring.length());
		    memcpy ((void *) terminal.data (), msgstring.c_str(), msgstring.length());

		    socket2.send (terminal);
		    socket2.recv (&reply);//get thanks

		    //send getview MASH
		    zmq::message_t getview (8);
		    memcpy ((void *) getview.data (), "GET_VIEW", 8);
		    socket.send (getview);

		    //receive frame MASH
	   	    socket.recv (&reply); // 
		    replyframe = string(static_cast<char*>(reply.data()), reply.size());

		    //send getaction MASH
		    zmq::message_t getaction (10);
		    memcpy ((void *) getaction.data (), "GET_ACTION", 10);
		    socket.send (getaction);

		    //receive  action MASH
	   	    socket.recv (&reply); // 
		    replyaction = string(static_cast<char*>(reply.data()), reply.size());



///////////////////////////////////////////////////////////////////////////////////////////////
//				MAKE PREDICTION
///////////////////////////////////////////////////////////////////////////////////////////////	

			// send frame to prediction server DQN
			msgstring = replyframe;
			zmq::message_t sendframe (msgstring.length());
			memcpy ((void *) sendframe.data (), msgstring.c_str(), msgstring.length());

			socket2.send (sendframe);


			//get thanks DQN
			//zmq::message_t reply;
			socket2.recv (&reply);

			// send action to prediction server DQN
			//cout<<replyaction<<endl;
			msgstring = replyaction;
			zmq::message_t sendaction (msgstring.length());
			memcpy ((void *) sendaction.data (), msgstring.c_str(), msgstring.length());

			socket2.send (sendaction);


			//get action DQN
			//zmq::message_t reply;
			socket2.recv (&reply);

			predictionStr = string(static_cast<char*>(reply.data()), reply.size());

	
			
///////////////////////////////////////////////////////////////////////////////////////////////
//                      GET REWARD FROM MASH
	
			float accReward=0;
			for(int i=0;i<frame_skip;i++)
			{
			//send action
			zmq::message_t actionmsg (predictionStr.length());
			memcpy ((void *) actionmsg.data (), predictionStr.c_str(), predictionStr.length());
			socket.send (actionmsg);

			//receive suggested action
			//cout<< "getsuggestion" <<endl;
			socket.recv (&reply); // 
			replystr = string(static_cast<char*>(reply.data()), reply.size());
			//cout<<"suggested action  "<<replystr<<endl;
			//get reward
			//cout<< "getreward" <<endl;
			zmq::message_t getreward (10);
			memcpy ((void *) getreward.data (), "GET_REWARD", 10);
			socket.send (getreward);
			
			//receive reward
			//cout<< "recreward" <<endl;
			socket.recv (&reply); // 
			reward_string = string(static_cast<char*>(reply.data()), reply.size());

			//cout<<"reward_string = " << reward_string<< endl;
			accReward+= atof(reward_string.c_str());
			//cout<<"accReward = " << accReward<< endl;		
			}

			reward_string = Convert(accReward);

///////////////////////////////////////////////////////////////////////////////////////////////
//       			 send reward to DQN

			

			StringBuffer rsb;
			Writer<StringBuffer> rwriter(rsb);

			rwriter.StartObject();
			rwriter.String("A");
			rwriter.StartArray();

			rwriter.String(reward_string);
			
			rwriter.EndArray();
			rwriter.EndObject();

			rmessage= rsb.GetString();
								
			zmq::message_t reward_msg (rmessage.length());
			memcpy (reward_msg.data (), rsb.GetString(), rmessage.length());

			socket2.send(reward_msg);

			//socket2.recv (&reply); // dummy reply

////////////////////////////////////////////////////////////////////////////////////////////////
			//receive if timed out DQN
			//zmq::message_t reply;
			socket2.recv (&reply);

			replystr = string(static_cast<char*>(reply.data()), reply.size());
		        if(replystr == "Done")	
		        {
				cout<<"DONE"<<endl;
				roundended=1;
				break;
		        }
			else if(replystr == "NOTDone")
			{// Do nothing
				//cout<<"NOTDone"<<endl;
			}
			else
			{
				cout<<"ERROR: DONE = "<<replystr<<endl;
			}
//////////////////////////////////////////////////////////////////////////////////////////////

			//is round finished/failed?
			//cout<< "getreward" <<endl;
			zmq::message_t getstatus (10);
			memcpy ((void *) getstatus.data (), "GET_STATUS", 10);
			socket.send (getstatus);
			
			//receive status
			//cout<< "recreward" <<endl;
			socket.recv (&reply); // 
			replystr = string(static_cast<char*>(reply.data()), reply.size());

			counter++;
		}//End of one round ///////////////////////////////////////////////

		if(replystr=="FINISHED")
			numFinish++;

		result <<i<<","<< numFinish << endl;
		//cout<<"numfinished  = "<<numFinish<<endl;
		//resettask
		zmq::message_t resettask (5);
		memcpy ((void *) resettask.data (), "RESET", 5);
		socket.send (resettask);
		
		//receive OK
		socket.recv (&reply); // 
		replystr = string(static_cast<char*>(reply.data()), reply.size());

		///////send terminal to RL server///////////////////////////////////
		if(roundended){

			msgstring = "NOTTERMINAL";
			zmq::message_t endterminal (msgstring.length());
			memcpy ((void *) endterminal.data (), msgstring.c_str(), msgstring.length());

			socket2.send (endterminal);

			//get thanks
			socket2.recv (&reply);

			//continue;
		}
		else{
			msgstring = "TERMINAL";
			zmq::message_t endterminal (msgstring.length());
			memcpy ((void *) endterminal.data (), msgstring.c_str(), msgstring.length());

			socket2.send (endterminal);

			//get thanks
			socket2.recv (&reply);
		}

		////////////////////////////////////////////

	}// end of all rounds
	

	result.close();
	//c->sendCommand("DONE",empty); // shut down connection to server
	//c->waitResponse(strResponse, arguments);

	cout<<"score = "<<numFinish<<endl;

	return 0;
}
