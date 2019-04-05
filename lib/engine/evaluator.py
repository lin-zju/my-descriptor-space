import torch
from torch.nn import DataParallel
from tqdm import tqdm

def evaluate(model, device, dataloader, evaluator):
    """
    Test engine.
    
    :param model: model(data, targets) -> results
    :param dataloader: next(iter(dataloader)) -> data, targets
    :param evaluator: evaluator.evaluate(data, targets, results),
                      evaluator.clear()
                      evaluator.get_results() -> results
    :return: the results returned by evaluator
    """
    
    cpu = torch.device('cpu')
    model = model.to(device)
    if isinstance(model, DataParallel):
        model = model.module
        
    with torch.no_grad():
        model.eval()
        pbar = tqdm(dataloader)
        for (data, targets) in pbar:
            data = {k: v.to(device) for (k, v) in data.items()}
            targets = {k: v.to(device) for (k, v) in targets.items()}
            results = model(data)
            evaluator.evaluate(data, targets, results)
            pbar.set_description('pck: {:.3f}'.format(evaluator.average_precision()))
            # evaluator.average_precision()
            
    print('Final pck: {:.3f}'.format(evaluator.get_results()))
        
    return evaluator.get_results()
    
    
