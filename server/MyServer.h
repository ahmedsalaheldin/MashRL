#include <SimulationServer.h>
#include <mash-appserver/application_server_interface.h>
#include <zmq.hpp>


using namespace Mash;
using namespace std;

typedef std::map<std::string, Mash::ArgumentsList>  tTaskParametersList;

class MyServer
{

public:
	MyServer();
	~MyServer();

public:

	std::string Receive();
	void Send(string message);
	void Loop();
	void Handlemessage(string message);
	string Convert (float number);


private:
	IApplicationServer*             _pApplicationServer;
        tTaskParametersList             _taskParameters;
	zmq::context_t* context;
	zmq::socket_t* socket;
	float reward;
	bool bFinished;
	bool bFailed;
	string strEvent;
};
