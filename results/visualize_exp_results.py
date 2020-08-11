
def visualize_avg_rew(infilepath, outfilepath):
    """
    :param infilepath: the name of the file from which to read the results of the experiment
    :param outfilepath: where to save the figure generated
    :return:
    """
    exp_res = open(infilepath, "r")
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    ax = plt.subplot(111)

    for mdp in exp_res:
        #splitting on double quotes
        mdp = mdp.split("\"")

        # if ground, first list item will have the word "ground"
        if ("ground" in mdp[0]):
            #and will contain everything we need as a comma seperated string
            mdp = mdp[0].split(",")
        else:
            #if not, the name of the abstraction will be the second list item
            #and everything else we need will be in the 3rd list item
            #which needs to be cleaned of empty strings
            mdp = [mdp[1]] + [m for m in mdp[2].split(",") if m != ""]

        print(mdp)

        episodes = [i for i in range(1, len(mdp))]
        plt.plot(episodes, [float(i) for i in mdp[1:]], label="%s" % (mdp[0],))

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.savefig(outfilepath)
visualize_avg_rew("exp_output.csv","exp_results_plot")