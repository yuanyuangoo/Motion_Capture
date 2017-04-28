#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <caffe/util/signal_handler.h>
#include <caffe/caffe.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using std::ostringstream;
int train()
{
	vector<string> stages=get_stage_from_flag();
	caffe::SolverParameter solver_param;
	caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver,&solver_param);
	solver_param.mutable_tarin_state()->set_level(FLAGS_level);
	for(int i=0;i<stages.size();i++){
		solver_param.mutable_tarin_state->add_stage(stages[i]);
	}
	if (FLAGS_gpu.size()==0
			&& solver_param.has_solver_mode()
			&& solver_param.solver_mode()==caffe::SolverParameter_SolverMode_GPU) {
		if (solver_param.has_device_id()) {
			FLAGS_gpu="" +
				boost::lexical_cast<string>(solver_param.device_id());
		} else {
			FLAGS_gpu= ""+ boost::lexical_cast<string>(0);
		}
	}
	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size()==0) {
		LOG(INFO)<< "Use CPU.";
	} else {
		ostringstream s;
		for (int i=0;i<gpus.size(); ++i)
		{
			s<<(i ? ", " : "") << gpus[i];
		}
		LOG(INFO)<<"Using GPUs "<<s.str();
		solver_param.set_device_id(gpus[0]);
		Caffe::setDevice(gpus[0]);
		Caffe::set_mode(Caffe::GPU);
		Caffe::set_solver_count(gpus.size());
	}
	caffe::SignalHandler_signal_handler(
			GetRequestedAction(FLAGS_sigint_effect),
			GetRequestedAction(FLAGS_sighup_effect));

	shared_ptr<caffe::Solver<float>>
		solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	solver->SetActionFunction(signal_handler.GetActionFunction());

	if(FLAGS_snapshot.size()) {
		LOG(INFO) <<"Resuming from"<< FLAGS_snapshot;
		solver->Restore(FLAGS_snapshot.c_str());
	} else if(FLAGS_weights.size()){
		CopyLayers(solver.get(),FLAGS_weights);
	}
	LOG(INFO)<<"Staring Optimization";
	if (gpus.size()>1)
	{

	}

}
typedef int (*BrewFunction());
/*static void get_gpus(vector<int>* gpus)
{
	if(FLAGES_gpu=="all")
	{
		int count=0;
	}
}
*/
static BrewFunction GetBrewFunction(const caffe::string& name){
	if(g_brew_map.count(name))
	{
		return g_brew_map[name];
	}
	else
	{
		LOG(ERROR)<<"Available caffe actions:";
		for (BrewMap::iter it=g_brew_map.begin();
				it!=g_brew_map.end(); ++it)
		{
			LOG(ERROR)<<"\t"<<it->first;
		}
		LOG(FATAL)<<"Unknown action: "<< name;
		return NULL;
	}
}
int main(int argc,char** argv)
{
	FLAGS_alsologtostderr=1;
	gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
	caffe::GlobalInit(&argc,&argv);
	if (argc==2)
		return GetBrewFunction(caffe::string(argv[1]))();
	return 0;
}
