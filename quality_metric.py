import numpy as np
def score(y_trues, y_preds):
    '''
    Версия 1.3:

    На вход подаются многомерные списки y_trues, y_preds

    Для каждой манипуляции в каждом тексте в y_trues ищем расстояние до ближайшей манипуляции
    в соответствующем тексте y_preds (пока без учёта длины пересечения)

    Находим среднее расстояние по тексту и записываем в список scores

    Возвращаем среднее по всем текстам
    '''
    # esxception на равную размерность массивов
    if len(y_trues) != len(y_preds):
        print('Error1: Количество текстов не совпадает')
        return None
    
    scores = []
    for z in range(len(y_trues)):
        y_true = y_trues[z]
        y_pred = y_preds[z]
        if 1 in y_true and 1 not in y_pred:
            continue
        if len(y_true) != len(y_pred):
            print(f'Error2: Количество слов в текстах с индексом {z} не совпадает')
            return None
        # список вхождений манипуляций
        manip_list = [] # список [[начало, конец], [], [], ...] манипуляций
        for i in range(len(y_true)):
            if y_true[i] == 1:
                for j in range(i + 1, len(y_true)):
                    if y_true[j] != 2:
                        manip_list.append([i, j])
                        break
        if y_true[i] == 1:
            manip_list.append([i, i + 1])

        dists = [] # список расстояний до болижайшего (левого, если расстояние одинаково) начала манипуляции
        n = -1
        for i, j in manip_list:
            n += 1
            dists.append(len(y_pred)) 
            for k in range(len(y_pred) - dists[n], len(y_pred)):
                if y_pred[k] == 1 and abs(i-k) < dists[n]:
                    dists[n] = abs(i-k)

        scores.append(sum(dists)/len(dists))
        
        # стараемся минимизировать метрику
    return (sum(scores)/len(scores))
