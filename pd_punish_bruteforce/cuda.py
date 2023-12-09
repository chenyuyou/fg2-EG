
output_status = r"""
FLAMEGPU_AGENT_FUNCTION(output_status, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    const flamegpu::id_t id = FLAMEGPU->getID();    
    float score = FLAMEGPU->getVariable<float>("score");
    unsigned int move = FLAMEGPU->getVariable<unsigned int>("move");

//    printf("%d\n", static_cast<int>(move));

    FLAMEGPU->message_out.setVariable<int>("id", id);
    FLAMEGPU->message_out.setVariable<float>("score", score);
    FLAMEGPU->message_out.setVariable<unsigned int>("move", move);
    return flamegpu::ALIVE;
}
"""

study= r"""
FLAMEGPU_AGENT_FUNCTION(study, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    float score = FLAMEGPU->getVariable<float>("score");
    unsigned int move = FLAMEGPU->getVariable<unsigned int>("move");
    unsigned int next_move = FLAMEGPU->getVariable<unsigned int>("next_move");
    int num_agents = FLAMEGPU->environment.getProperty<unsigned int>("num_agents");
    int selected = FLAMEGPU->random.uniform<int>(1, num_agents+1);
    
    move = next_move;

    for (const auto& msg : FLAMEGPU->message_in) {
        if (msg.getVariable<int>("id") == selected) {
            const float other_score = msg.getVariable<float>("score");
            const unsigned int other_move = msg.getVariable<unsigned int>("move");

            const float intense = FLAMEGPU->environment.getProperty<float>("intense");
            float k = FLAMEGPU->environment.getProperty<float>("k");
            float c = FLAMEGPU->environment.getProperty<float>("c");
            float f = FLAMEGPU->environment.getProperty<float>("f");
            const float noise = FLAMEGPU->environment.getProperty<float>("noise");
            auto payoff = FLAMEGPU->environment.getMacroProperty<float,3,3>("payoff");
//            printf("%d\n", static_cast<int>(payoff[1][2]));
            
            if (score < other_score){
                next_move = other_move;
            } else {
                next_move = move;
            }
            if (move == 0) {
                f = 0.0;
                k = 0.0;
            } else if (move == 1){
                c = 0.0;
                f = 0.0;
                k = 0.0;
            } else {
                c = 0.0;
            }

            float pay_off = expf(intense * payoff[move][other_move] * (1 - expf(-(noise + ((1 - noise) * (k * f + c))))));
            score = score + pay_off;
        }
    }

//    float rng = FLAMEGPU->random.uniform<float>();
//    const float mu = FLAMEGPU->environment.getProperty<float>("mu");


//    if (rng < mu) {
//        unsigned int selected_move = FLAMEGPU->random.uniform<unsigned int>(0, 2);    
//        next_move = selected_move;
//    } 



    FLAMEGPU->setVariable<float>("score", score);
    FLAMEGPU->setVariable<unsigned int>("move", move);
    FLAMEGPU->setVariable<unsigned int>("next_move", next_move);


    return flamegpu::ALIVE;
}
"""

mutate = r"""
FLAMEGPU_AGENT_FUNCTION(mutate, flamegpu::MessageNone, flamegpu::MessageNone) {
    float rng = FLAMEGPU->random.uniform<float>();
    const float mu = FLAMEGPU->environment.getProperty<float>("mu");
    unsigned int move = FLAMEGPU->getVariable<unsigned int>("move");

    if (rng < mu) {
        unsigned int selected = FLAMEGPU->random.uniform<unsigned int>(0, 2);    
        FLAMEGPU->setVariable<unsigned int>("next_move", selected);
    } 
    return flamegpu::ALIVE;
}
"""