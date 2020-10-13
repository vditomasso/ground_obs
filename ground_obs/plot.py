import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

def subplot(rv_df, obj_name, preferred_range=False, frac=True):

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(2,1,sharex=True,figsize=[8, 10])

    plt.subplots_adjust(wspace=0, hspace=0)

    if preferred_range != False:
        # ASSUMING THAT 2 THINGS IN PREFERRED RANGE LIST MEANS VELOCITY PREFERENCE, 1 THING MEANS BLENDED FRACTION
        if len(preferred_range)==2:
        # Find where the RV is in the preferable range
            indices1 = np.where((rv_df['rv'].values < -np.min(preferred_range)) & (rv_df['rv'].values > -np.max(preferred_range)))[0]
            indices2 = np.where((rv_df['rv'].values > np.min(preferred_range)) & (rv_df['rv'].values < np.max(preferred_range)))[0]
            indices = np.concatenate([indices1,indices2])
            ax[0].scatter(rv_df['date'][indices].values, rv_df['rv'][indices].values, alpha=0.2, s=50, c='orange',
                      label=fr'RV +- {np.min(preferred_range)}-{np.max(preferred_range)} km/s')

        elif len(preferred_range)==1:
            # Find where the blended fraction is in the preferable range
            indices = np.where(rv_df['blend_frac'].values < preferred_range[0])[0]
            ax[1].scatter(rv_df['date'][indices].values, rv_df['blend_frac'][indices].values, alpha=0.2, s=50, c='orange',
                      label=fr'Blended fraction < {np.min(preferred_range)}')
            ax[0].scatter(rv_df['date'][indices].values, rv_df['rv'][indices].values, alpha=0.2, s=50, c='orange')

    ax[0].set_title(fr'{obj_name}', fontsize=18)

    indices_split = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

    # Find the indicies that define the border between favorable RV and disfavorable RV
    key_indices = []
    for list_element in indices_split:
        if len(list_element) > 5:
            key_indices.append(np.min(list_element))
            key_indices.append(np.max(list_element))

    # Set timeticks locations and label for plotting
    timeticks_locations = rv_df['date'][key_indices].values
    timeticks_labels = []
    for key_index in key_indices:
        time_it = Time(rv_df['date'][key_index],format='jd')
        timeticks_labels.append(time_it.to_value('iso').split()[0])

    ax[0].plot(rv_df['date'].values,rv_df['rv'].values)
    ax[1].plot(rv_df['date'].values,rv_df['blend_frac'].values)

    for key_index in key_indices:
        ax[0].axvline(rv_df['date'][key_index], linestyle='--', alpha=0.5)
        ax[1].axvline(rv_df['date'][key_index], linestyle='--', alpha=0.5)

    # Mark the date ranges for preferable RVs
    ax[1].set_xticks(timeticks_locations)
    ax[1].set_xticklabels(timeticks_labels, rotation=45)
    ax[1].set_xlabel('Times (year-month-day)', fontsize=14)
    ax[0].set_ylabel('RV Earth-Exoplanet (km/s)', fontsize=14)
    ax[1].set_ylabel('Fraction of Blended Lines', fontsize=14)

    plt.legend(loc='upper left')

    plt.show()

def rv_curve(rv_df, obj_name, preferred_range=False, frac=True):
    '''This is only half written'''

    fig, ax = plt.subplots(figsize=[8, 6])

    if preferred_range != False:
        # ASSUMING THAT 2 THINGS IN PREFERRED RANGE LIST MEANS VELOCITY PREFERENCE, 1 THING MEANS BLENDED FRACTION
        if len(preferred_range)==2:
        # Find where the RV is in the preferable range
            indices1 = np.where((rv_df['rv'].values < -np.min(preferred_range)) & (rv_df['rv'].values > -np.max(preferred_range)))[0]
            indices2 = np.where((rv_df['rv'].values > np.min(preferred_range)) & (rv_df['rv'].values < np.max(preferred_range)))[0]
            indices = np.concatenate([indices1,indices2])

        elif len(preferred_range)==1:
            # Find where the blended fraction is in the preferable range
            indices = np.where(rv_df['blend_frac'].values < preferred_range[0])[0]

    # Highlight areas of the curve with preferable RVs
    ax.scatter(rv_df['date'][indices].values, rv_df['rv'][indices].values, alpha=0.2, s=50, c='orange',
                label=fr'RV +- {np.min(preferred_range)}-{np.max(preferred_range)} km/s')
    ax.set_title(fr'{obj_name}', fontsize=18)

    indices_split = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

    # Find the indicies that define the border between favorable RV and disfavorable RV
    key_indices = []
    for list_element in indices_split:
        if len(list_element) > 5:
            key_indices.append(np.min(list_element))
            key_indices.append(np.max(list_element))

    # Set timeticks locations and label for plotting
    timeticks_locations = rv_df['date'][key_indices].values
    timeticks_labels = []
    for key_index in key_indices:
        time_it = Time(rv_df['date'][key_index],format='jd')
        timeticks_labels.append(time_it.to_value('iso').split()[0])

    # plt.plot(times.value, v_star)
    ax.plot(rv_df['date'].values,rv_df['rv'].values)

    # Mark the date ranges for preferable RVs
    ax.set_xticks(timeticks_locations)
    ax.set_xticklabels(timeticks_labels, rotation=45)
    ax.set_xlabel('Times (year-month-day)', fontsize=14)
    ax.set_ylabel('RV Earth-Exoplanet (km/s)', fontsize=14)

    # Draw lines that mark the preferable date ranges
    for key_index in key_indices:
        ax.axvline(rv_df['date'][key_index], linestyle='--', alpha=0.5)

    if frac==True:
        ax2 = ax.twinx()
        ax2.set_ylabel('Blended Fraction')
        ax2.plot(rv_df['date'].values,rv_df['blend_frac'].values)

    # plt.legend(fontsize=14)

    plt.show()