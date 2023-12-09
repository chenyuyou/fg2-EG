import pyflamegpu
import sys, random, math, time
import matplotlib.pyplot as plt
from cuda import *

def create_model():
    model = pyflamegpu.ModelDescription("punish")
    return model

def define_environment(model):
    """
        Environment
    """
    env = model.Environment()

    env.newPropertyUInt("num_agents", 10000)
    env.newPropertyUInt("cooperation", 0)
    env.newPropertyUInt("defect", 0)
    env.newPropertyUInt("punishment", 0)

    env.newPropertyFloat("intense", 0.1)

    env.newPropertyFloat("k", 6.0)

    env.newPropertyFloat("b", 16.0)
    env.newPropertyFloat("c", 15.0)
    env.newPropertyFloat("e", 5.0)
    env.newPropertyFloat("f", 5.0)

    env.newPropertyFloat("noise", 0.1)
    env.newPropertyFloat("mu", 0.01)

    env.newMacroPropertyFloat("payoff",3,3)


def define_messages(model):
    message = model.newMessageBruteForce("status_message")
    message.newVariableID("id")
    message.newVariableFloat("score")
    message.newVariableUInt("move")   

def define_agents(model):
    # Create the agent
    agent = model.newAgent("agent")
    # Assign its variables
    agent.newVariableFloat("score")
    agent.newVariableUInt("move")
    agent.newVariableUInt("next_move")    
 
    # Assign its functions
    fn = agent.newRTCFunction("output_status", output_status)
    fn.setMessageOutput("status_message")

    fn = agent.newRTCFunction("study", study)
    fn.setMessageInput("status_message")

    fn = agent.newRTCFunction("mutate", mutate)
#    fn.setMessageInput("predator_location_message")
#    fn.setMessageOutput("prey_eaten_message")
#    fn.setMessageOutputOptional(True)
#    fn.setAllowAgentDeath(True)
        
def define_execution_order(model):
    """
      Control flow
    """    
    layer = model.newLayer()
    layer.addAgentFunction("agent", "output_status")

    layer = model.newLayer()
    layer.addAgentFunction("agent", "study")

    layer = model.newLayer()
    layer.addAgentFunction("agent", "mutate")
    
    model.addInitFunction(initfn())
    model.addStepFunction(stepfn())



class stepfn(pyflamegpu.HostFunction):        
    def run(self, FLAMEGPU):
        agents = FLAMEGPU.agent("agent")
        cooperation = agents.countUInt("move", 0)
        defect = agents.countUInt("move", 1)
        punishment = agents.countUInt("move", 2)
        FLAMEGPU.environment.setPropertyUInt("cooperation", cooperation)
        FLAMEGPU.environment.setPropertyUInt("defect", defect)
        FLAMEGPU.environment.setPropertyUInt("punishment", punishment)


class initfn(pyflamegpu.HostFunction):        
    def run(self, FLAMEGPU):
        payoff = FLAMEGPU.environment.getMacroPropertyFloat("payoff")
        b = FLAMEGPU.environment.getPropertyFloat("b")
        c = FLAMEGPU.environment.getPropertyFloat("c")
        e = FLAMEGPU.environment.getPropertyFloat("e")
        f = FLAMEGPU.environment.getPropertyFloat("f")
        payoff[0][0]=b-c
        payoff[0][1]=-c
        payoff[0][2]=-c-e
        payoff[1][0]=b
        payoff[1][1]=0
        payoff[1][2]=-e
        payoff[2][0]=b-f
        payoff[2][1]=-f
        payoff[2][2]=-f-e

        num_agents = FLAMEGPU.environment.getPropertyUInt("num_agents")
        agents = FLAMEGPU.agent("agent")
        for i in range(num_agents):            
            agent = agents.newAgent()
            agent.setVariableFloat("score", 0)
            agent.setVariableUInt("move", random.choice([0,1,2]))
            agent.setVariableUInt("next_move", random.choice([0,1,2]))

def define_logs(model):
    log = pyflamegpu.StepLoggingConfig(model)
    log.setFrequency(10)
#    log.logEnvironment("REPRODUCE_PREY_PROB")
    log.logEnvironment("cooperation")
    log.logEnvironment("defect")
    log.logEnvironment("punishment")

    return log

core=100

def define_output(ensemble):
    ensemble.Config().out_directory = "results"
    ensemble.Config().out_format = "json"
#    ensemble.Config().concurrent_runs = 1
    ensemble.Config().timing = True
    ensemble.Config().truncate_log_files = True
    ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast
    ensemble.Config().devices = pyflamegpu.IntSet([0])

def define_runs(model):
    ## 设置为要测试的参数。
    runs = pyflamegpu.RunPlanVector(model, 16)
    runs.setSteps(100000)
    runs.setRandomSimulationSeed(12, 1)
    runs.setPropertyLerpRangeFloat("c", 0.0, 15.0)
    return runs

def initialise_simulation(seed):
    model = create_model()
    define_messages(model)
    define_agents(model)
    define_environment(model)
    define_execution_order(model)
    runs = define_runs(model)
    logs = define_logs(model)
    ensembleSimulation = pyflamegpu.CUDAEnsemble(model)
    define_output(ensembleSimulation)
    ensembleSimulation.setStepLog(logs)
    ensembleSimulation.simulate(runs)


if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()