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

    env.newPropertyUInt("num_agents", 100)
    env.newPropertyUInt("cooperation", 0)
    env.newPropertyUInt("defect", 0)
    env.newPropertyUInt("punishment", 0)

    env.newPropertyFloat("intense", 1.0)

    env.newPropertyFloat("k", 6.0)

    env.newPropertyFloat("b", 16.0)
    env.newPropertyFloat("c", 15.0)
    env.newPropertyFloat("e", 5.0)
    env.newPropertyFloat("f", 5.0)

    env.newPropertyFloat("noise", 0.1)
    env.newPropertyFloat("mu", 0.0001)

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

    fn = agent.newRTCFunction("output_status", output_status)
    fn.setMessageOutput("status_message")

    fn = agent.newRTCFunction("study", study)
    fn.setMessageInput("status_message")

    fn = agent.newRTCFunction("mutate", mutate)

        
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

def define_logs(model):
    log = pyflamegpu.StepLoggingConfig(model)
    log.setFrequency(1)
#    log.agent("agent").logCount()
    log.logEnvironment("cooperation")
    log.logEnvironment("defect")
    log.logEnvironment("punishment")
#    log.logEnvironment("REPRODUCE_PREY_PROB")
    return log

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
#        payoff=((b-c, -c, -c-e),
#                ( b,   0,   -e),
#                (b-f, -f, -f-e))  
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
        
#        agents = FLAMEGPU.agent("agent","cooperation")
        for i in range(num_agents):            
            agent = agents.newAgent()
            agent.setVariableFloat("score", 0)
            agent.setVariableUInt("move", random.choice([0,1,2]))
            agent.setVariableUInt("next_move", random.choice([0,1,2]))
        


def define_runs(model):
    ## 设置为要测试的参数。
    runs = pyflamegpu.RunPlan(model)
    runs.setRandomSimulationSeed(123456)
    runs.setSteps(10000)
#    runs.setPropertyLerpRangeFloat("REPRODUCE_PREY_PROB", 0.05, 1.05)
    return runs

def initialise_simulation(seed):
    model = create_model()
    define_messages(model)
    define_agents(model)
    define_environment(model)
    define_execution_order(model)
    logs = define_logs(model)
    run=define_runs(model)
    cuda_sim = pyflamegpu.CUDASimulation(model)
    cuda_sim.setStepLog(logs)
    cuda_sim.simulate(run)
    cuda_sim.exportLog(
  "log.json", # The file to output (must end '.json' or '.xml')
  True,       # Whether the step log should be included in the log file
  False,       # Whether the exit log should be included in the log file
  False,       # Whether the step time should be included in the log file (treated as false if step log not included)
  False,       # Whether the simulation time should be included in the log file (treated as false if exit log not included)
  False) 



if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()