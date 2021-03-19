from experiment import *

if __name__ == "__main__":
    # Hyperparameter setting
    json_path = os.path.join("experiments", "params.json")
    assert os.path.isfile(json_path), "No JSON configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Experiments test
    # exp1 = Experiments(exp_name="baseline_standalone_resnet18", params=params)
    # exp1.set_baseline_dataset()
    # exp1.train_baseline_standalone()

    # Experiments test2
    # exp2 = Experiments(exp_name="baseline_standalone_alexnet", params=params)
    # exp2.set_baseline_dataset()
    # exp2.train_baseline_standalone()

    # Experiments KD (ResNet -> AlexNet)
    # exp3 = Experiments(exp_name="baseline_KD_res2alex", params=params)
    # exp3.set_baseline_dataset()
    # exp3.train_baseline_KD()

    # Experiments KD (ResNet -> AlexNet), alpha = 0.0, 0.1, 0.2, ..., 1.0
    # exp4_1 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.0", params=params)
    # exp4_1.set_baseline_dataset()
    # exp4_1.train_baseline_KD()
    #
    # del exp4_1
    #
    # params.alpha = 0.1
    # exp4_2 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.1", params=params)
    # exp4_2.set_baseline_dataset()
    # exp4_2.train_baseline_KD()
    #
    # del exp4_2
    #
    # params.alpha = 0.2
    # exp4_3 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.2", params=params)
    # exp4_3.set_baseline_dataset()
    # exp4_3.train_baseline_KD()
    #
    # del exp4_3
    #
    # params.alpha = 0.3
    # exp4_4 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.3", params=params)
    # exp4_4.set_baseline_dataset()
    # exp4_4.train_baseline_KD()
    #
    # del exp4_4
    #
    # params.alpha = 0.4
    # exp4_5 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.4", params=params)
    # exp4_5.set_baseline_dataset()
    # exp4_5.train_baseline_KD()
    #
    # del exp4_5
    #
    # params.alpha = 0.5
    # exp4_6 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.5", params=params)
    # exp4_6.set_baseline_dataset()
    # exp4_6.train_baseline_KD()
    #
    # del exp4_6
    #
    # params.alpha = 0.6
    # exp4_7 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.6", params=params)
    # exp4_7.set_baseline_dataset()
    # exp4_7.train_baseline_KD()
    #
    # del exp4_7
    #
    # params.alpha = 0.7
    # exp4_8 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.7", params=params)
    # exp4_8.set_baseline_dataset()
    # exp4_8.train_baseline_KD()
    #
    # del exp4_8
    #
    # params.alpha = 0.8
    # exp4_9 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.8", params=params)
    # exp4_9.set_baseline_dataset()
    # exp4_9.train_baseline_KD()
    #
    # del exp4_9
    #
    # params.alpha = 0.9
    # exp4_10 = Experiments(exp_name="baseline_KD_res2alex_alpha_0.9", params=params)
    # exp4_10.set_baseline_dataset()
    # exp4_10.train_baseline_KD()
    #
    # del exp4_10
    #
    # params.alpha = 1.0
    # exp4_11 = Experiments(exp_name="baseline_KD_res2alex_alpha_1.0", params=params)
    # exp4_11.set_baseline_dataset()
    # exp4_11.train_baseline_KD()
    #
    # del exp4_11

    for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
        params.temperature = t
        exp5 = Experiments(exp_name="baseline_KD_res2alex_T_" + str(t), params=params)
        exp5.set_baseline_dataset()
        exp5.train_baseline_KD()

        del exp5