import pyflamegpu
import sys, random, math, time
import matplotlib.pyplot as plt
from cuda import *

# 1. 定义一个类来封装模拟参数
class EnsembleParameters:
    def __init__(self,
                 num_agents=10000,
                 intense=1.0,
                 k=0.0,
                 b=16.0,
                 c=15.0,
                 e=0.0,
                 f=0.0,
                 noise=0.1,
                 mu=0.0001,
                 simulation_steps=10000,
                 num_runs=12,  # 新增参数：集成模拟的次数
                 random_seed_start=12, # 新增参数：随机种子起始值
                 random_seed_increment=1, # 新增参数：随机种子增量
                 output_directory="results", # 新增参数：输出目录
                 output_format="json", # 新增参数：输出格式
                 concurrent_runs=None, # 新增参数：并发运行数 (None表示由FLAMEGPU决定)
                 devices=[0] # 新增参数：使用的GPU设备
                 ):
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
        self.num_runs = num_runs
        self.random_seed_start = random_seed_start
        self.random_seed_increment = random_seed_increment
        self.output_directory = output_directory
        self.output_format = output_format
        self.concurrent_runs = concurrent_runs
        self.devices = devices


def create_model():
    model = pyflamegpu.ModelDescription("punish")
    return model

# 2. 修改 define_environment 函数，使其接受参数对象
def define_environment(model, params: EnsembleParameters):
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
    # 如果不注册stepfn函数，环境中的变量值将不会改变，始终为初始值
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
    log.setFrequency(1)    # 设置日志记录频率
#    log.logEnvironment("REPRODUCE_PREY_PROB")
    log.logEnvironment("cooperation")
    log.logEnvironment("defect")
    log.logEnvironment("punishment")

    return log


# 3. 修改 define_output 函数，使其接受参数对象
def define_output(ensemble, params: EnsembleParameters):
    # 意味着所有模拟运行的输出文件将被保存在当前工作目录下名为 "results" 的文件夹中
    ensemble.Config().out_directory = params.output_directory
    # 这指定了模拟的输出日志文件将使用 JSON 格式进行保存
    ensemble.Config().out_format = params.output_format
    # 设置并发运行数
    if params.concurrent_runs is not None:
        ensemble.Config().concurrent_runs = params.concurrent_runs
    # 框架可能会记录和报告模拟运行所花费的时间
    ensemble.Config().timing = True
    # 那么在开始新的模拟运行时，旧的日志文件将被清空（截断）而不是追加内容或报错
    ensemble.Config().truncate_log_files = True
    # 这通常意味着选择一个较低的错误检查级别，可能以牺牲一些调试能力为代价来换取更快的执行速度
    ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast
    # 即使系统中有多个 GPU，也只使用第一个 GPU (通常是设备 0)
    ensemble.Config().devices = pyflamegpu.IntSet(params.devices)

# 4. 修改 define_runs 函数，使其接受参数对象
def define_runs(model, params: EnsembleParameters):
    ## 设置为要测试的参数。
    # 在这里，它创建了一个包含单个运行计划的向量。每个运行计划定义了一次模拟运行的具体配置。
    runs = pyflamegpu.RunPlanVector(model, params.num_runs)
    # 设置此向量中所有运行计划的模拟步数
    runs.setSteps(params.simulation_steps) # 使用参数设置模拟步数
    # 设置此向量中运行计划的随机数生成器种子。
    # - 第一个参数 '12' 是第一个运行计划的基础种子 (initial seed)。
    # - 第二个参数 '1' 是种子增量 (seed increment)。
    # 对于向量中的第 i 个运行计划（从 0 开始计数），其实际种子将是 initial_seed + i * seed_increment。
    # 因为这里只有一个运行计划 (i=0)，所以它的随机种子将被设置为 12 + 0 * 1 = 12。
    # 这对于确保模拟的可复现性或进行随机性研究很重要。
    runs.setRandomSimulationSeed(params.random_seed_start, params.random_seed_increment) # 使用参数设置随机种子
    # 这通常用于参数扫描：如果你创建了多个运行计划（例如，在 RunPlanVector 构造函数中设置数量大于 1），
    # 这个函数会让属性 "c" 的值在不同的运行计划中从 0.0（第一个计划）线性插值（Lerp）到 15.0（最后一个计划）。
    # 因为当前只有一个运行计划，并且此行被注释，所以属性 "c" 将使用模型中定义的默认值。
#    runs.setPropertyLerpRangeFloat("c", 0.0, 15.0)
    return runs

# 5. 修改 initialise_simulation 函数，创建参数对象并传递
def initialise_simulation(params: EnsembleParameters):
    model = create_model()
    define_messages(model)
    define_agents(model)
    define_environment(model, params) # 传递参数对象
    define_execution_order(model)
    # 3. 定义运行计划
    # 调用 define_runs(model)。此函数创建一个或多个运行计划（RunPlan），通常打包在 RunPlanVector 中。
    # 每个运行计划指定一次模拟运行的具体参数，如运行的总步数、随机种子、
    # 以及（如果进行参数扫描）该次运行的环境属性值。
    # 返回的 RunPlanVector 存储在 'runs' 变量中。
    runs = define_runs(model, params) # 传递参数对象
    # 4. 定义日志记录
    # 调用 define_logs(model)。此函数定义在模拟运行时需要记录哪些数据。
    # 这可能包括环境属性的值、特定代理变量的聚合值（如平均值、总和）等。
    # 通常会返回一个配置好的日志对象（如 StepLog 或 ExitLog）。
    # 返回的日志配置存储在 'logs' 变量中。
    logs = define_logs(model)
    # 5. 创建模拟执行器 (Ensemble)
    # 使用配置完整的 'model' 对象创建一个 pyflamegpu.CUDAEnsemble 实例。
    # CUDAEnsemble 是负责在 GPU 上管理和执行一个或多个模拟运行（即一个集成）的核心对象。
    ensembleSimulation = pyflamegpu.CUDAEnsemble(model)
    # 6. 配置输出
    # 调用 define_output(ensembleSimulation)。此函数配置模拟集成的输出设置，
    # 例如输出文件的目录、格式、是否记录时间、使用的 GPU 设备等。
    define_output(ensembleSimulation, params) # 传递参数对象
    # 7. 关联日志配置
    # 将步骤日志记录配置（'logs' 对象）与模拟集成关联起来。
    # 这告诉 'ensembleSimulation' 在每个模拟步骤结束时需要记录 'logs' 中定义的数据。
    ensembleSimulation.setStepLog(logs)
    ensembleSimulation.simulate(runs)


if __name__ == "__main__":
    start = time.time()
    # 6. 在这里实例化 EnsembleParameters 并调整参数
    ensemble_params = EnsembleParameters(
        num_agents=20000,       # 更改智能体数量
        simulation_steps=5000, # 更改模拟步数
        num_runs=20,           # 更改集成模拟次数
        output_directory="ensemble_results" # 更改输出目录
        # 其他参数会使用默认值，如果你想修改其他参数，在这里添加即可
    )
    initialise_simulation(ensemble_params) # 将参数对象传递给初始化函数
    end = time.time()
    print(f"Ensemble simulation finished in {end - start:.2f} seconds.") # 格式化输出时间
    exit()