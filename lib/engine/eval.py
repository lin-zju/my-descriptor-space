import torch

def test(model, device, dataloader, evaluator):
    """
    Test engine.
    
    :param model: model(data, targets) -> results
    :param dataloader: next(iter(dataloader)) -> data, targets
    :param evaluator: evaluator.evaluate(data, targets, results),
                      evaluator.clear()
                      evaluator.get_results() -> results
    :return: the results returned by evaluator
    """
    
    model = model.to(device)
    model.eval()
    for (data, targets) in dataloader:
        data = {k: v.to(device) for (k, v) in data.items()}
        targets = {k: v.to(device) for (k, v) in targets.items()}
        results = model.inference(data)
        evaluator.evaluate(data, targets, results)
        
    return evaluator.results()
    
    
