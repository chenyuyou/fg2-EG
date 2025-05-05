import pyflamegpu
import sys, random, math, time
import matplotlib.pyplot as plt
from cuda import *
import datetime

# 1. 定义一个类来封装模拟参数
class SimulationParameters:
    def __init__(self,
                 num_agents=100,
                 intense=1.0,
                 k=6.0,
                 b=16.0,
                 c=15.0,
                 e=5.0,
                 f=5.0,
                 noise=0.1,
                 mu=0.0001,
                 simulation_steps=2000,
                 random_seed=123456):
        self.num_agents = num_agents
        self.intense = intense
        self.k = k
        self.b = b
        self.c = c
        self.e = e
        self.f = f
        self.noise = noise
        self.mu = mu
        self.simulation_steps = simulation_steps
        self.random_seed = random_seed

def create_model():
    model = pyflamegpu.ModelDescription("punish")
    return model

# 2. 修改 define_environment 函数，使其接受参数对象
def define_environment(model, params: SimulationParameters):
    """
        Environment
    """
    env = model.Environment()

    env.newPropertyUInt("num_agents", params.num_agents)
    env.newPropertyUInt("cooperation", 0)
    env.newPropertyUInt("defect", 0)
    env.newPropertyUInt("punishment", 0)

    env.newPropertyFloat("intense", params.intense)

    env.newPropertyFloat("k", params.k)

    env.newPropertyFloat("b", params.b)
    env.newPropertyFloat("c", params.c)
    env.newPropertyFloat("e", params.e)
    env.newPropertyFloat("f", params.f)

    env.newPropertyFloat("noise", params.noise)
    env.newPropertyFloat("mu", params.mu)

    env.newMacroPropertyFloat("payoff", 3, 3)


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

    # 将 addInitFunction 和 addStepFunction 放在 define_execution_order 函数的任何位置（只要在定义完相应的函数对象之后）都是可以的，并且它们会在 FLAMEGPU 框架控制下的正确时机被执行。
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
    # 模拟的每个时间步结束时，统计当前模拟状态下所有智能体采取合作、背叛和惩罚这三种行为的数量，并将这些数量作为环境的共享属性进行更新
    # 获取这种全局状态信息
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
        payoff[0][0] = b - c
        payoff[0][1] = -c
        payoff[0][2] = -c - e
        payoff[1][0] = b
        payoff[1][1] = 0
        payoff[1][2] = -e
        payoff[2][0] = b - f
        payoff[2][1] = -f
        payoff[2][2] = -f - e

        num_agents = FLAMEGPU.environment.getPropertyUInt("num_agents")
        agents = FLAMEGPU.agent("agent")

#        agents = FLAMEGPU.agent("agent","cooperation")
        for i in range(num_agents):
            agent = agents.newAgent()
            agent.setVariableFloat("score", 0)
            agent.setVariableUInt("move", random.choice([0, 1, 2]))
            agent.setVariableUInt("next_move", random.choice([0, 1, 2]))


# 3. 修改 define_runs 函数，使其接受参数对象
def define_runs(model, params: SimulationParameters):
    ## 设置为要测试的参数。
    runs = pyflamegpu.RunPlan(model)
    runs.setRandomSimulationSeed(params.random_seed)
    runs.setSteps(params.simulation_steps)
#    runs.setPropertyLerpRangeFloat("REPRODUCE_PREY_PROB", 0.05, 1.05)
    return runs

# 4. 修改 initialise_simulation 函数，创建参数对象并传递
def initialise_simulation(params: SimulationParameters):
    model = create_model()
    define_messages(model)
    define_agents(model)
    define_environment(model, params)  # 传递参数对象
    define_execution_order(model)
    logs = define_logs(model)
    run = define_runs(model, params)  # 传递参数对象
    cuda_sim = pyflamegpu.CUDASimulation(model)
    cuda_sim.setStepLog(logs)
    cuda_sim.simulate(run)
    # --- 开始修改 ---
    # 2. 获取当前时间
    now = datetime.datetime.now()
    # 3. 格式化时间字符串 (YYYYMMDDHHMM)
    timestamp_str = now.strftime("%Y%m%d%H%M")
    # 4. 构造完整的文件名
    log_filename = f"log{timestamp_str}.json"

    cuda_sim.exportLog(
        log_filename,  # The file to output (must end '.json' or '.xml')
        True,       # Whether the step log should be included in the log file
        False,       # Whether the exit log should be included in the log file
        False,       # Whether the step time should be included in the log file (treated as false if step log not included)
        False,       # Whether the simulation time should be included in the log file (treated as false if exit log not included)
        False)


if __name__ == "__main__":
    start = time.time()

    # 5. 在这里实例化 SimulationParameters 并调整参数
    simulation_params = SimulationParameters(
        num_agents=200,       # 更改智能体数量
        intense=1,          # 更改 Intense 参数
        k=6.0,
        b=16.0,
        c=15.0,
        e=5.0,
        f=5.0,
        noise=0.1,
        mu=0.0001,
        simulation_steps=2000,
        random_seed=123456# 其他参数会使用默认值，如果你想修改其他参数，在这里添加即可
    )

    initialise_simulation(simulation_params) # 将参数对象传递给初始化函数

    end = time.time()
    print(f"Simulation finished in {end - start:.2f} seconds.") # 格式化输出时间
    exit()


