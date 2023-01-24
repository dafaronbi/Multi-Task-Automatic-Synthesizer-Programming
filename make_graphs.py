import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import sys
import numpy as np
import os
import tensorflow as tf
import model
import scipy.stats
from data import one_hot
from data import parameter_label

def stat_eval():
    # multi_df = pd.read_csv(sys.argv[1])

    # serum_df = multi_df[multi_df['synth'] == "serum"]
    # diva_df = multi_df[multi_df['synth'] == "diva"]
    # tyrell_df = multi_df[multi_df['synth'] == "tyrell"]
    # pd.set_option('display.max_columns', 500)
    # print("<=========== ALL ===========>")
    # print(multi_df.describe())
    # print("<=========== SERUM ===========>")
    # print(serum_df.describe())
    # print("<=========== DIVA ===========>")
    # print(diva_df.describe())
    # print("<=========== TYRELL ===========>")
    # print(tyrell_df.describe())

    serum_df = pd.read_csv("serum_metrics.csv")
    diva_df = pd.read_csv("diva_metrics.csv")
    tyrell_df = pd.read_csv("tyrell_metrics.csv")

    separate_separate_df = pd.concat([serum_df,diva_df,tyrell_df])
    joint_separate_df = pd.read_csv("multi_metrics_64.csv")
    joint_joint_df = pd.read_csv("single_metrics.csv")

    # print(separate_separate_df["lsd"].to_numpy())
    F, p = scipy.stats.f_oneway(separate_separate_df["lsd"].to_numpy(), joint_separate_df["lsd"].to_numpy(), joint_joint_df["lsd"].to_numpy())

    print(F)
    print(p)

    s, p = scipy.stats.ttest_ind(separate_separate_df["lsd"].to_numpy(), joint_separate_df["lsd"].to_numpy())

    print(s)
    print(p)

    s, p = scipy.stats.ttest_ind(joint_joint_df["lsd"].to_numpy(), joint_separate_df["lsd"].to_numpy())

    print(s)
    print(p)

    s, p = scipy.stats.ttest_ind(separate_separate_df["lsd"].to_numpy(), joint_joint_df["lsd"].to_numpy())

    print(s)
    print(p)


def csv_eval():
    # multi_df = pd.read_csv(sys.argv[1])

    # serum_df = multi_df[multi_df['synth'] == "serum"]
    # diva_df = multi_df[multi_df['synth'] == "diva"]
    # tyrell_df = multi_df[multi_df['synth'] == "tyrell"]
    # pd.set_option('display.max_columns', 500)
    # print("<=========== ALL ===========>")
    # print(multi_df.describe())
    # print("<=========== SERUM ===========>")
    # print(serum_df.describe())
    # print("<=========== DIVA ===========>")
    # print(diva_df.describe())
    # print("<=========== TYRELL ===========>")
    # print(tyrell_df.describe())

    serum_df = pd.read_csv("serum_metrics.csv")
    diva_df = pd.read_csv("diva_metrics.csv")
    tyrell_df = pd.read_csv("tyrell_metrics.csv")

    all_df = pd.concat([serum_df,diva_df,tyrell_df])
    print(all_df.describe())

def hpss_boxplot():
    multi_df = pd.read_csv(sys.argv[1])

    lsd_data = []
    for hpss in [20,40,60,80,100]:
        lsd_data.append(multi_df[multi_df['hpss'] == hpss]['lsd'].to_numpy())

    plt.boxplot(lsd_data, labels= ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"], showfliers=False)
    plt.ylabel("Log Spectral Distance")
    plt.xlabel("Harmonic Component Percentage")
    plt.yscale('log')
    plt.savefig("hpss_box.pdf")

def latent_size():
    lsd =[]
    mse = []
    acc = []

    lsd_err = []
    mse_err = []
    acc_err = []

    sizes = [2,4,8,16,32,64,128,256,512]

    for l_size in sizes:
        df = pd.read_csv("multi_metrics_" + str(l_size) + ".csv")

        lsd.append(df["lsd"].mean())
        mse.append(df["con_mse"].mean())
        acc.append(df["class_accuracy"].mean())

        lsd_err.append(df["lsd"].std())
        mse_err.append(df["con_mse"].std())
        acc_err.append(df["class_accuracy"].std())

    fig, ax = plt.subplots(3, sharex=True)

    # axs[0].boxplot(, labels=sizes, showfliers=False, showmeans=True)
    eb0 = ax[0].errorbar(sizes, lsd, lsd_err, linestyle='--', marker='^', capsize = 3)
    eb1 = ax[1].errorbar(sizes, mse, mse_err, linestyle='--', marker='^', color='red', capsize = 3)
    eb2 = ax[2].errorbar(sizes, acc, acc_err, linestyle='--', marker='^', color='green', capsize = 3)
    eb0[-1][0].set_linestyle(':')
    eb1[-1][0].set_linestyle(':')
    eb2[-1][0].set_linestyle(':')
    ax[0].set_xscale('log')
    ax[0].set_xticks(sizes)
    ax[0].get_xaxis().set_major_formatter(mtick.ScalarFormatter())
    ax[0].set_ylabel("LSD")
    ax[2].set_xlabel("Latent Size")
    ax[1].set_ylabel("Continuous MSE")
    ax[2].set_ylabel("Categorical % ACC")
    ax[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    n_lsd = (lsd - np.min(lsd))/ (np.max(lsd) - np.min(lsd))
    n_mse = (mse - np.min(mse))/ (np.max(mse) - np.min(mse))
    n_acc = (acc - np.min(acc))/ (np.max(acc) - np.min(acc))
    print(np.diff(n_lsd))
    print(np.diff(n_mse))
    print(np.diff(n_acc))

    for j in [0,1,2]:
        ax[j].yaxis.set_label_coords(-0.13, 0.5)

    fig.tight_layout()
    plt.savefig("latent_size.pdf")

def latent_control():

    print("Loading Data...")
    test_names = np.load("test_name.npy")
    test_spec_data = np.load("test_mels.npy",allow_pickle=True)
    test_serum_params = np.load("test_serum_params.npy",allow_pickle=True)
    test_serum_masks = np.load("test_serum_mask.npy",allow_pickle=True)
    test_diva_params = np.load("test_diva_params.npy",allow_pickle=True)
    test_diva_masks = np.load("test_diva_mask.npy",allow_pickle=True)
    test_tyrell_params = np.load("test_tyrell_params.npy",allow_pickle=True)
    test_tyrell_masks = np.load("test_tyrell_mask.npy",allow_pickle=True)
    test_h_labels = np.load("test_hpss.npy",allow_pickle=True)
    test_synth = np.load("test_synth.npy",allow_pickle=True)
    s_attack = np.load("data generation/s_attack_con.npy",allow_pickle=True)
    d_attack = np.load("data generation/d_attack_con.npy",allow_pickle=True)
    t_attack = np.load("data generation/t_attack_con.npy",allow_pickle=True)
    s_release = np.load("data generation/s_release_con.npy",allow_pickle=True)
    d_release = np.load("data generation/d_release_con.npy",allow_pickle=True)
    t_release = np.load("data generation/t_release_con.npy",allow_pickle=True)
    s_lowpass = np.load("data generation/s_lowpass_con.npy",allow_pickle=True)
    d_lowpass = np.load("data generation/d_lowpass_con.npy",allow_pickle=True)
    t_lowpass = np.load("data generation/t_lowpass_con.npy",allow_pickle=True)
    s_highpass = np.load("data generation/s_highpass_con.npy",allow_pickle=True)
    d_highpass = np.load("data generation/d_highpass_con.npy",allow_pickle=True)
    t_highpass = np.load("data generation/t_highpass_con.npy",allow_pickle=True)
    print("Done!")

    m_size = len(test_spec_data)

    #define shapes
    l_dim = 64
    i_dim = (1, 128, 431, 1)

    #directory for finding checkpoints
    checkpoint_path = "saved_models/vst_multi_64/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #get latest model
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    #batch_size
    batch_size = 32

    #number of batches in one epoch
    batches_epoch = m_size // batch_size

    #warmup amounta
    warmup_it = 100*batches_epoch

    #create model
    m = model.vae_multi(64, i_dim, test_serum_params.shape[-1], test_diva_params.shape[-1], test_tyrell_params.shape[-1], model.optimizer, warmup_it)

    #load stored weights
    m.load_weights(latest)

    l_model = tf.keras.Model(inputs=m.get_layer("input_7").input, outputs=m.get_layer("sampling_2").output)

    z_s_attack = l_model.predict(s_attack)
    z_d_attack = l_model.predict(d_attack)
    z_t_attack = l_model.predict(t_attack)

    z_s_release = l_model.predict(s_release)
    z_d_release = l_model.predict(d_release)
    z_t_release = l_model.predict(t_release)

    z_s_lowpass = l_model.predict(s_lowpass)
    z_d_lowpass = l_model.predict(d_lowpass)
    z_t_lowpass = l_model.predict(t_lowpass)

    z_s_highpass = l_model.predict(s_highpass)
    z_d_highpass = l_model.predict(d_highpass)
    z_t_highpass = l_model.predict(t_highpass)

    a_r_matrix = np.zeros((3,64))
    r_r_matrix = np.zeros((3,64))
    lp_r_matrix = np.zeros((3,64))
    hp_r_matrix = np.zeros((3,64))

    for z_num in range(64):
        x = np.arange(0,1,0.01)
        a_r_matrix[0][z_num] =scipy.stats.pearsonr(x,z_s_attack[:,z_num])[0]
        a_r_matrix[1][z_num] =scipy.stats.pearsonr(x,z_d_attack[:,z_num])[0]
        a_r_matrix[2][z_num]=scipy.stats.pearsonr(x,z_t_attack[:,z_num])[0]

        r_r_matrix[0][z_num] =scipy.stats.pearsonr(x,z_s_release[:,z_num])[0]
        r_r_matrix[1][z_num] =scipy.stats.pearsonr(x,z_d_release[:,z_num])[0]
        r_r_matrix[2][z_num]=scipy.stats.pearsonr(x,z_t_release[:,z_num])[0]

        lp_r_matrix[0][z_num] =scipy.stats.pearsonr(x,z_s_lowpass[:,z_num])[0]
        lp_r_matrix[1][z_num] =scipy.stats.pearsonr(x,z_d_lowpass[:,z_num])[0]
        lp_r_matrix[2][z_num]=scipy.stats.pearsonr(x,z_t_lowpass[:,z_num])[0]

        hp_r_matrix[0][z_num] =scipy.stats.pearsonr(x,z_s_highpass[:,z_num])[0]
        hp_r_matrix[1][z_num] =scipy.stats.pearsonr(x,z_d_highpass[:,z_num])[0]
        hp_r_matrix[2][z_num]=scipy.stats.pearsonr(x,z_t_highpass[:,z_num])[0]
    
    za_sum = np.round(np.abs(np.sum(a_r_matrix, axis=0)),1)
    zr_sum = np.round(np.abs(np.sum(r_r_matrix, axis=0)),1)
    zlp_sum = np.round(np.abs(np.sum(lp_r_matrix, axis=0)),1)
    zhp_sum = np.round(np.abs(np.sum(hp_r_matrix, axis=0)),1)

    print("ATTACK")
    print(np.argmax(za_sum))
    print(np.max(za_sum))
    print("RELEASE")
    print(np.argmax(zr_sum))
    print(np.max(zr_sum))
    print("LOW PASS")
    print(np.argmax(zlp_sum))
    print(np.max(zlp_sum))
    print("HIGHPASS")
    print(np.argmax(zhp_sum))
    print(np.max(zhp_sum))

    fig, ax = plt.subplots(2,2)
    x_trace = np.arange(0,1,0.01)
    ax[0][0].figure.set_size_inches(15,15)
    ax[0][0].scatter(x_trace, z_s_attack[:,53], label="serum")
    ax[0][0].scatter(x_trace, z_d_attack[:,53], label="diva")
    ax[0][0].scatter(x_trace, z_t_attack[:,53], label="tyrell")
    ax[0][0].set_xlabel("Attack Norm")
    ax[0][0].set_ylabel("Z Value")
    ax[0][0].legend()
    ax[0][0].set_title("Z53 vs Attack")

    ax[0][1].figure.set_size_inches(15,15)
    ax[0][1].scatter(x_trace, z_s_release[:,51], label="serum")
    ax[0][1].scatter(x_trace, z_d_release[:,51], label="diva")
    ax[0][1].scatter(x_trace, z_t_release[:,51], label="tyrell")
    ax[0][1].set_xlabel("Release Norm")
    ax[0][1].set_ylabel("Z Value")
    ax[0][1].legend()
    ax[0][1].set_title("Z51 vs Release")

    ax[1][0].figure.set_size_inches(15,15)
    ax[1][0].scatter(x_trace, z_s_lowpass[:,14], label="serum")
    ax[1][0].scatter(x_trace, z_d_lowpass[:,14], label="diva")
    ax[1][0].scatter(x_trace, z_t_lowpass[:,14], label="tyrell")
    ax[1][0].set_xlabel("Lowpass Cuttoff")
    ax[1][0].set_ylabel("Z Value")
    ax[1][0].legend()
    ax[1][0].set_title("Z14 vs Lowpass Cuttoff")

    ax[1][1].figure.set_size_inches(15,15)
    ax[1][1].scatter(x_trace, z_s_highpass[:,15], label="serum")
    ax[1][1].scatter(x_trace, z_d_highpass[:,15], label="diva")
    ax[1][1].scatter(x_trace, z_t_highpass[:,15], label="tyrell")
    ax[1][1].set_xlabel("Highpass Cuttoff")
    ax[1][1].set_ylabel("Z Value")
    ax[1][1].legend()
    ax[1][1].set_title("Z54 vs Highpass Cuttoff")

    plt.savefig("latent_correlations.png")

def group_plot():
    multi_df = pd.read_csv("group_metrics.csv")

    fig, axs = plt.subplots( 2,1, figsize=(17, 20), sharex=True)

    con_df = multi_df.loc[:,["vol_con", "pitch_con", "lfo_con", "env_con", "mod_con", "osc_con", "fil_con", "fx_con"]]
    class_df = multi_df.loc[:,["vol_class", "pitch_class", "lfo_class", "env_class", "mod_class", "osc_class", "fil_class", "fx_class"]]
    all_df = multi_df.loc[:,["vol_all", "pitch_all", "lfo_all", "env_all", "mod_all", "osc_all", "fil_all", "fx_all"]]

    print(con_df.to_numpy())
    print(class_df.to_numpy())

    con_df = con_df[['vol_con', 'pitch_con', 'env_con', 'fil_con', 'lfo_con', 'mod_con', 'osc_con', 'fx_con']]
    class_df = class_df[['vol_class', 'pitch_class', 'env_class', 'fil_class', 'lfo_class', 'mod_class', 'osc_class', 'fx_class']]

    print(con_df)
    print(class_df)

    print(con_df.std())
    print(class_df.std())

    print(con_df.mean())
    print(class_df.mean())

    c_list = ['pink', 'lightblue', 'lightgreen', 'tomato', 'bisque', 'darkviolet', 'paleturquoise', 'crimson']
    n_parameters = [21, 22, 63, 72, 88, 103, 128, 191]
    n_cat_parameters = [3, 6, 9, 23, 6, 5, 52, 41]
    n_con_parameters = [18, 16, 54, 49, 82, 98, 76, 150]

    bp0 = axs[0].boxplot(class_df.to_numpy(), labels=["volume", "pitch", "envelope", "filter", "lfo", "mod", "oscillator", "fx"], showfliers=False, showmeans=True)
    axs[0].set_ylabel('Class Accuracy')
    bp1 = axs[1].boxplot(con_df.to_numpy(), labels=["volume", "pitch", "envelope", "filter", "lfo", "mod", "oscillator", "fx"], showfliers=False, showmeans=True)
    axs[1].set_ylabel('Continuous Mean Square Error')
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axs[0].figure.set_size_inches(10, 7)
    axs[1].figure.set_size_inches(10, 7)
    
    fig.subplots_adjust(hspace=0.2)

    for bplot in (bp0, bp1):
        for patch,color in zip(bplot['boxes'], c_list):
            patch.set(color=color)

    counts = np.random.randint(0, 25, len(axs[1].get_xticks()))
    print(axs[1].get_xticks())
    for i, xpos in enumerate(axs[1].get_xticks()[:8]):
        axs[0].text(xpos,-0.2, str(n_cat_parameters[i]), 
                size = 10, ha = 'center')

        axs[1].text(xpos,-3, str(n_con_parameters[i]), 
                size = 10, ha = 'center')

    axs[0].text(0,-0.2, "# of Parameters:", size = 10, ha = 'center')
    axs[1].text(0,-3, "# of Parameters:", size = 10, ha = 'center')

    axs[0].legend([bp0['medians'][0], bp0['means'][0]], ['median', 'mean'])

    # axs[0].
    plt.savefig("group_plots.pdf")

    # print(all_df.to_numpy())

    # con_val = con_df.mean().sort_values(ascending=False).values
    # class_val = class_df.mean().sort_values(ascending=True).values
    # all_val = all_df.mean().sort_values(ascending=False).values
    
    # # print(a_multi_df.index[:8])
    # axs[0].bar(["lfo", "fx", "fil", "mod", "vol", "osc", "pitch", "env"], con_val, 0.35, 
    # color = ['orange', 'cyan', 'red', 'purple', 'blue', 'brown', 'green', 'pink'])
    # axs[0].set_ylabel('Scores')
    # axs[0].set_title('Continuous MSE')
    # axs[0].legend()

    # ["vol", "pitch", "fil", "mod", "env", "osc", "fx", "lfo"]
    # # print(d_multi_df.index[1:9])
    # axs[1].bar(["vol", "pitch", "fil", "mod", "env", "osc", "fx", "lfo"], class_val, 0.35, 
    # color = ['blue', 'green', 'red', 'purple', 'pink', 'brown', 'cyan', 'orange'])
    # axs[1].set_ylabel('Scores')
    # axs[1].set_title('Class Accuracy')
    # axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # ["fx", "fil", "vol", "lfo", "osc", "mod", "env", "pitch"]
    # # print(d_multi_df.index[1:9])
    # axs[2].bar(["fx", "fil", "vol", "lfo", "osc", "mod", "env", "pitch"], all_val, 0.35, 
    # color = ['cyan', 'red', 'blue', 'orange', 'brown', 'purple', 'pink', 'green'])
    # axs[2].set_ylabel('Scores')
    # axs[2].set_title('All MSE')

    # add_value_labels(axs[0])
    # add_value_labels(axs[1], l_type="percent")
    # add_value_labels(axs[2])

    # fig.tight_layout()

    # plt.savefig("group_plot.pdf")

def add_value_labels(ax, spacing=5, l_type=None):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if l_type == "percent":
            label = "{:.1f}%".format(y_value*100)

        else:
            label = "{:.3f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

def ood_confuse():
    multi_df = pd.read_csv("ood_metrics.csv")

    confuse = np.zeros((3,3))

    confuse[0] = [multi_df[multi_df['synth'] == 'serum'].mean().values[2], multi_df[multi_df['synth'] == 'serum'].mean().values[5],multi_df[multi_df['synth'] == 'serum'].mean().values[8]]
    confuse[1] = [multi_df[multi_df['synth'] == 'diva'].mean().values[2], multi_df[multi_df['synth'] == 'diva'].mean().values[5],multi_df[multi_df['synth'] == 'diva'].mean().values[8]]
    confuse[2] = [multi_df[multi_df['synth'] == 'tyrell'].mean().values[2], multi_df[multi_df['synth'] == 'tyrell'].mean().values[5],multi_df[multi_df['synth'] == 'tyrell'].mean().values[8]]

    labels = ["Serum","Diva","Tyrell"]
    plt.matshow(confuse, cmap="Blues")
    plt.xticks([0,1,2],labels)
    plt.yticks([0,1,2],labels)
    plt.colorbar(label="Log Spectral Distance Value", location = 'bottom')
    plt.xlabel("Ground Truth VST")
    plt.ylabel("Generated VST")
    plt.title("Out of domain audio comparison (Lower is more similar)")
    plt.savefig("ood.pdf", bbox_inches='tight', pad_inches=1)
    
    print(confuse)

def count_params():
    serum_con = {"osc": 0, "vol": 0, "env": 0, "pitch": 0, 
    "fil": 0, "fx": 0, "mod": 0, "lfo": 0}

    serum_cat = {"osc": 0, "vol": 0, "env": 0, "pitch": 0, 
    "fil": 0, "fx": 0, "mod": 0, "lfo": 0}

    diva_con = {"osc": 0, "vol": 0, "env": 0, "pitch": 0, 
    "fil": 0, "fx": 0, "mod": 0, "lfo": 0}

    diva_cat = {"osc": 0, "vol": 0, "env": 0, "pitch": 0, 
    "fil": 0, "fx": 0, "mod": 0, "lfo": 0}

    tyrell_con = {"osc": 0, "vol": 0, "env": 0, "pitch": 0, 
    "fil": 0, "fx": 0, "mod": 0, "lfo": 0}

    tyrell_cat = {"osc": 0, "vol": 0, "env": 0, "pitch": 0, 
    "fil": 0, "fx": 0, "mod": 0, "lfo": 0}

    all_con = {"vol": 0, "pitch": 0, "env": 0, "fil": 0, 
    "lfo": 0, "mod": 0, "osc": 0, "fx": 0}

    all_cat = {"vol": 0, "pitch": 0, "env": 0, "fil": 0, 
    "lfo": 0, "mod": 0, "osc": 0, "fx": 0}

    for i,label in enumerate(parameter_label.serum_labels):
        if(one_hot.serum_oh[i] > 0):
            serum_cat[label] += 1
        else:
            serum_con[label] += 1

    for i,label in enumerate(parameter_label.diva_labels):
        if(one_hot.diva_oh[i] > 0):
            diva_cat[label] += 1
        else:
            diva_con[label] += 1

    for i,label in enumerate(parameter_label.tyrell_labels):
        if(one_hot.serum_oh[i] > 0):
            tyrell_cat[label] += 1
        else:
            tyrell_con[label] += 1

    for key in ["osc", "vol", "env", "pitch", 
    "fil", "fx", "mod", "lfo"]:
        all_con[key] = serum_con[key] + diva_con[key] + tyrell_con[key]
        all_cat[key] = serum_cat[key] + diva_cat[key] + tyrell_cat[key]

    print("serum_con")
    print(serum_con)
    print("serum_cat")
    print(serum_cat)
    print("diva_con")
    print(diva_con)
    print("diva_cat")
    print(diva_cat)
    print("tyrell_con")
    print(tyrell_con)
    print("tyrell_cat")
    print(tyrell_cat)
    print("all_con")
    print(all_con)
    print("all_cat")
    print(all_cat)
        

def main():
    stat_eval()
    # csv_eval()
    # hpss_boxplot()
    # latent_size()
    # latent_control()
    # group_plot()
    # ood_confuse()
    # count_params()

if __name__ == "__main__":
    main()