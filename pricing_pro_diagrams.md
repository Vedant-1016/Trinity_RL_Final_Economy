# Pricing Pro: Advanced System Architecture

Here is the finalized workflow diagram for your pitch deck. It clearly illustrates the separation between the OpenEnv Environment we built, and the Reinforcement Learning (RL) training loop that your teammate will implement.

```mermaid
graph TD
    %% Style Definitions
    classDef env fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D47A1
    classDef agent fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20
    classDef action fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100
    classDef rl fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C

    subgraph RLLoop ["Reinforcement Learning (RL) Training Loop"]
        direction TB
        
        LLM["AI-CFO (The LLM)<br/>(The Agent Being Trained)"]:::agent
        
        subgraph Env ["Pricing Pro Environment (The Gym)"]
            direction TB
            STATE["State Generator<br/>(Inventory + 9 Params including:<br/>Price Sensitivity & Urgency)"]:::env
            COMP["Competitor Agent<br/>(Calculates Rival Prices)"]:::env
            CONS["Consumer Council<br/>(Evaluates Brand vs Price)"]:::env
            CALC["Environment Engine<br/>(Calculates Sales & Profit)"]:::env
            
            STATE --> COMP
            COMP --> CONS
            CONS --> CALC
        end
        
        ACTION["Action<br/>(JSON: Price array for inventory)"]:::action
        UPDATE["RL Algorithm (TRL / GRPO)<br/>(Updates LLM Neural Weights)"]:::rl
        
        %% Workflow Connections
        STATE -.->|1. Observes Market Context| LLM
        LLM -->|2. Chooses Pricing Strategy| ACTION
        ACTION ===>|3. Inputs Prices into Env| COMP
        
        CALC -.->|4. Returns Reward ($ Profit)| UPDATE
        UPDATE ===>|5. Modifies LLM Brain| LLM
        
        %% Multi-Turn Loop
        CALC -.->|Loops for 7 Days| STATE
    end
```

> **How to pitch this diagram:**
> *"Judges, the blue box is the OpenEnv environment we built. It contains the 9 macro parameters and internal competitor/consumer agents. The green box is the external LLM. As you can see, the purple RL algorithm acts as the bridge—taking the profit reward from our environment and using it to mathematically train the LLM to become a pricing genius."*
