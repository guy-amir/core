from random_forest import WaveletsForestRegressor

def calc_smoothness(x, y):
    wfr = WaveletsForestRegressor(regressor='random_forest', criterion='mse', depth=9, trees=5)
    wfr.fit(x, y)
    alpha, n_wavelets, errors = wfr.evaluate_smoothness(m=1000)
    return alpha

def smoothness_layers(outputs,labels):  
    a = []
    l = labels.view(labels.size(0),1) #change shape from (4,) to (4,1)
    for layer in outputs:
        if len(layer.size())>2:
            layer = layer.view(layer.size(0),-1)
        a.append(calc_smoothness(layer.data.cpu().numpy(), l.data.cpu().numpy()))
    print(f'smoothness of layers: {a}')
    return a