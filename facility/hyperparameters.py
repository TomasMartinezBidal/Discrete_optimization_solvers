import pandas as pd

data = {'customers': [50, 200, 100, 1000, 800, 3000, 1500, 2000],
        'facilities': [25, 50, 100, 100, 200, 500, 1000, 2000],
        'solve_full': [True, True, True, True, False, False, False, False],
        'objective_threshold': [3269821.32053,
                                3732793.43377,
                                1965.55449699,
                                22724634,
                                4711295,
                                30000000,
                                10000000,
                                10000000],
        'time_limit': [60 * 60, 60 * 60, 60 * 60, 60 * 60, 60 * 60, 60 * 60, 60 * 60, 60 * 60],
        'search_till_opt': [True, True, True, False, False, False, False, False],
        'facilities_subset': [25,50,100,100,60,60,60,60]}

df_hyperparameters = pd.DataFrame(data).set_index(keys=['customers', 'facilities'])


def hellow():
    print('hellow')


if __name__ == '__main__':
    print('Hellow! world!')
