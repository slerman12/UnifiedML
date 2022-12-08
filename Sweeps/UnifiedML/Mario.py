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
    experiment=Self-Supervised_Mario 
    +agent.depth=3
    parallel=true
    """
]


runs.UnifiedML.plots = [
    ['Mario', 'Self-Supervised_Mario'],
]

runs.UnifiedML.bluehive = False
runs.UnifiedML.title = 'Mario'

