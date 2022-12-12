from Sweeps.Templates import template


runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    # Standard
    """
    task=mario 
    train_steps=4000000 
    save_per_steps=500000 
    Agent=Agents.AC2Agent 
    experiment=Mario
    """,

    # Self-Supervised
    """
    task=mario 
    train_steps=4000000 
    save_per_steps=500000 
    Agent=Agents.AC2Agent 
    experiment=Self-Supervised-Mario 
    +agent.depth=3
    parallel=true
    """
]


runs.UnifiedML.plots = [
    ['Mario', 'Self-Supervised-Mario', 'Self-Supervised-Mario-2'],
]

runs.UnifiedML.bluehive = False
runs.UnifiedML.title = 'Mario'

