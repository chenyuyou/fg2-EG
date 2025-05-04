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

    env.newPropertyFloat("intense", 1.0)

    env.newPropertyFloat("k", 0)

    env.newPropertyFloat("b", 16.0)
    env.newPropertyFloat("c", 15.0)
    env.newPropertyFloat("e", 0)
    env.newPropertyFloat("f", 0)

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

#core=100
# 这个 ensemble 对象很可能是一个 pyflamegpu.CUDAEnsemble 实例
def define_output(ensemble):
    # 意味着所有模拟运行的输出文件将被保存在当前工作目录下名为 "results" 的文件夹中
    ensemble.Config().out_directory = "results"
    # 这指定了模拟的输出日志文件将使用 JSON 格式进行保存
    ensemble.Config().out_format = "json"
#    ensemble.Config().concurrent_runs = 1
    # 框架可能会记录和报告模拟运行所花费的时间
    ensemble.Config().timing = True
    # 那么在开始新的模拟运行时，旧的日志文件将被清空（截断）而不是追加内容或报错
    ensemble.Config().truncate_log_files = True
    # 这通常意味着选择一个较低的错误检查级别，可能以牺牲一些调试能力为代价来换取更快的执行速度
    ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast
    # 即使系统中有多个 GPU，也只使用第一个 GPU (通常是设备 0)
    ensemble.Config().devices = pyflamegpu.IntSet([0])

def define_runs(model):
    ## 设置为要测试的参数。
    # 在这里，它创建了一个包含单个运行计划的向量。每个运行计划定义了一次模拟运行的具体配置。
    runs = pyflamegpu.RunPlanVector(model, 1)
    # 设置此向量中所有运行计划的模拟步数
    runs.setSteps(10000)
    # 设置此向量中运行计划的随机数生成器种子。
    # - 第一个参数 '12' 是第一个运行计划的基础种子 (initial seed)。
    # - 第二个参数 '1' 是种子增量 (seed increment)。
    # 对于向量中的第 i 个运行计划（从 0 开始计数），其实际种子将是 initial_seed + i * seed_increment。
    # 因为这里只有一个运行计划 (i=0)，所以它的随机种子将被设置为 12 + 0 * 1 = 12。
    # 这对于确保模拟的可复现性或进行随机性研究很重要。
    runs.setRandomSimulationSeed(12, 1)
    # 这通常用于参数扫描：如果你创建了多个运行计划（例如，在 RunPlanVector 构造函数中设置数量大于 1），
    # 这个函数会让属性 "c" 的值在不同的运行计划中从 0.0（第一个计划）线性插值（Lerp）到 15.0（最后一个计划）。
    # 因为当前只有一个运行计划，并且此行被注释，所以属性 "c" 将使用模型中定义的默认值。
#    runs.setPropertyLerpRangeFloat("c", 0.0, 15.0)
    return runs

def initialise_simulation(seed):
    model = create_model()
    define_messages(model)
    define_agents(model)
    define_environment(model)
    define_execution_order(model)
    # 3. 定义运行计划
    # 调用 define_runs(model)。此函数创建一个或多个运行计划（RunPlan），通常打包在 RunPlanVector 中。
    # 每个运行计划指定一次模拟运行的具体参数，如运行的总步数、随机种子、
    # 以及（如果进行参数扫描）该次运行的环境属性值。
    # 返回的 RunPlanVector 存储在 'runs' 变量中。
    runs = define_runs(model)
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
    define_output(ensembleSimulation)
    # 7. 关联日志配置
    # 将步骤日志记录配置（'logs' 对象）与模拟集成关联起来。
    # 这告诉 'ensembleSimulation' 在每个模拟步骤结束时需要记录 'logs' 中定义的数据。
    ensembleSimulation.setStepLog(logs)
    ensembleSimulation.simulate(runs)


if __name__ == "__main__":
    start=time.time()
    initialise_simulation(64)
    end=time.time()
    print(end-start)
    exit()