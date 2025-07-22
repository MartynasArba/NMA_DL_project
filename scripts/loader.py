import os
import sys
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

def get_bins(stim_time, bin_w = 10, start_adj = 500, end_adj = 2000):
    """
    Generates bins for firing rate calculation
    stim_time: int/float, stimulus time in seconds
    bin_w: bin width, ms
    start_adj: how much time before stim to include, ms
    end_adj: how much time after stim to include, ms
    """
    
    stim_time_ms = int(stim_time * 1000)
    start_ms = stim_time_ms - 500
    end_ms = stim_time_ms + 2000
    
    return np.arange(start_ms, end_ms, bin_w)

# download session data
def load_cache(data_path, filtered = True):
    """
    Loads AllenSDK Visual Coding cache and returns available session objects
    data_path: where to save data
    filtered: whether to select only males and wild-type animals (could be improved later)
    
    """


    manifest_path = os.path.join(data_path, 'manifest.json')
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    if filtered:
        criterion = ((sessions.full_genotype=='wt/wt')&(sessions.sex=='M'))&(sessions.ecephys_structure_acronyms.apply(lambda x: 'VISp' in x))
        sessions = sessions[criterion].sort_values(by = 'unit_count', ascending = False)
    return cache, sessions 

# donwload data for one selected session
def get_single_session(cache, sessions, selected_id, filter = 'VISp'):
    """
    Gets spike data (timestamps) of a single session
    cache: allenSDK cache object
    sessions: allenSDK session info object
    selected_id (int): which session to load
    filter (str, bool or None): brain area to filter units by, or None/False to not filter

    """
    session = cache.get_session_data(sessions.index.values[selected_id],
                                 isi_violations_maximum = np.inf,
                                 amplitude_cutoff_maximum = np.inf,
                                 presence_ratio_minimum = -np.inf,
                                 timeout = None)
    # load unit data
    units = session.units
    if filter:
        # filtering units
        units = units[(units.ecephys_structure_acronym == filter) & (units.isi_violations < 0.1)]

    return session, units

def get_stimuli(session):
    """
    Gets stimuli names that were presented >100 times and are not movies
    session: allenSDK session object
    """

    stim_cats = np.unique(session.stimulus_conditions.stimulus_name, return_counts =  True)
    stim_cats = stim_cats[0][(stim_cats[1]>100)&(['movie' not in x for x in stim_cats[0]])]
    stim_table = session.get_stimulus_table(stim_cats)
    return stim_cats, stim_table

def generate_firing_rates(session, stim_table, units, sort_by_rate = True):
    """
    Generates firing rate matrices of stim x unit x timebin for all passed units in a session, using all passed stimuli
    session: allenSDK session object
    stim_table: selected stimuli information
    units: allenSDK units object, will be used to get spikes
    sort_by_rate: whether to sort resulting cells by their global firing rate (all stim, all bins)
    """
    firing_rates_all_stim = []
    #likely inefficient
    for stim in tqdm.tqdm(stim_table.start_time.to_numpy()):
        bins = get_bins(stim) #timepoints of data
        #array for storing spike data around one stimulus (units x bins)
        firing_rates = np.zeros((len(units.index.values), bins.shape[0]+2))

        #loading spike times
        for i, unit in enumerate(units.index.values):
            all_spikes = session.spike_times[unit] #get all spikes of unit
            stim_spikes = all_spikes[(all_spikes >= stim - 0.5)&(all_spikes < stim + 2)]*1000 #convert to ms and filter
            rate = np.digitize(stim_spikes, bins) #digitize
            spikebin, count = np.unique(rate, return_counts = True) #get firing rates in each bin
            firing_rates[i][spikebin] = count
        firing_rates_all_stim.append(firing_rates)
        
    firing_rates_all_stim = np.array(firing_rates_all_stim) #convert to array
    # sort by total firing rate across all stimuli and time, which 
    if sort_by_rate:
        sort_idx = firing_rates_all_stim.sum(axis=(0, 2)).argsort()
    else:
        sort_idx = np.ones(len(firing_rates_all_stim[0, :, 0]), dtype=bool)

    return firing_rates_all_stim[:, sort_idx, :]

def save_data(x, y, fname, data_path):
    """
    Saves data to files
    fname: suffix to use
    data_path: folder in which to save
    
    """
    #dump x
    with open(f'{data_path}/{fname}_x.pkl', 'wb') as f:
        pickle.dump(x, file = f)
    #dump y
    with open(f'{data_path}/{fname}_y.pkl', 'wb') as f:
        pickle.dump(y, file = f)

    return None

if __name__ == '__main__':
    print('CLI support not available')
    pass
    #run all code
