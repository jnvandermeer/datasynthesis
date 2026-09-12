from pathlib import Path
import pandas as pd
from types import SimpleNamespace
from pudb.remote import set_trace
import numpy as np
from scramblers import phase_scramble
import logging
from tqdm import tqdm
import re
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# my logger!
logger = logging.getLogger(__name__)


# small helper function to map new to orig data...
def map_to_original(v, new_v):
    unique_v = np.unique(v)
    idx = np.abs(new_v[:, None] - unique_v).argmin(axis=1)
    return unique_v[idx]
    self.history = ''
    self.errorhistory = ''


class DataSynthesis(object):

    def __init__(self, seed=1, max_participants=1000, PCAthresh=0.70):
        self.data = None
        self.params = SimpleNamespace(
            participantid=None,
            data=None,
            type=None,
            unit=None,
            cols_to_keep=None,
            session=None,
            time_value=None,
            time_unit=None
        )
        self.all_set = False
        self.history = ''
        self.errorhistory = ''
        self.seed=seed
        self.max_participants = max_participants
        # the explained varaicne that we're going to reconstruct again:
        self.PCAthresh = PCAthresh

        # set_trace(port=4444, term_size=(140, 50))
        
        

    def load(self, filename, **params):
        """Load a CSV and associate parameters with it."""
        self.filename = filename
        self.dataframe = pd.read_csv(filename)

        for key, value in params.items():
            setattr(self.params, key, value)

        # we can also immediately make the empty dataframe.
        self.output = self.dataframe.reindex().astype(object)
        self.output.loc[:, :] = pd.NA

        required = ('participantid', 'session', 'data', 'type', 'unit')

        self.all_set = all(
            getattr(self.params, attr) is not None
            for attr in required
        )
        
        if self.all_set:
            logger.info('All column names seem to have been provided')
        else:
            logger.info('You need to enter column names with enterdata')
        
    

    def enterdata(self):
        """Check and interactively complete parameters."""

        for key, value in vars(self.params).items():

            if value is None:
                new_value = input(f'{key} is empty! Enter the new value: ')
                setattr(self.params, key, new_value)

            else:
                answer = input(
                    f'{key} is now: {value};   change it? (y/N): '
                )

                if answer.lower() == 'y':
                    new_value = input(f'Enter new value for {key}: ')
                    setattr(self.params, key, new_value)

        required = ('participantid', 'session', 'data', 'type', 'unit')

        self.all_set = all(
            getattr(self.params, attr) is not None
            for attr in required
        )

    def checkit(self):

        # once here, if all_set, we can find the things that need processing:
        if self.all_set:
    
            
            df=self.dataframe
            participants =  list(df[self.params.participantid].unique())
            print(f'There are {len(participants)} participants in this data sheet')
            for p in participants:
                print(f'\t{p}')
            sessions = list(df[self.params.session].unique())
            print(f'There are {len(sessions)} sessions in this data sheet')
            for s in sessions:
                print(f'\t{s}')

                if pd.isna(s):
                    print('There seems to be a NA in the period/session; I will tread NA as session "NA"')
            
            type_and_unit = df[[self.params.type, self.params.unit]].drop_duplicates().values.tolist()
            print('There are the following variables and units:')
            for t, u in type_and_unit:
                print(f'\t{t} -- in units: {u}')
    
        else:
            print('Some column names are not yet set! - call enterdata')

    

    
    def synthesize(self, seed=None):
        """Apply synthesis transformation."""

        if seed is not None:
            self.seed = seed
        
        df = self.dataframe
        participants = list(df[self.params.participantid].unique())

        width = len(str(self.max_participants))
        new_p_list = [f'P{i+1:0{width}d}' for i in range(self.max_participants)]
        # new_p_list = [f'P{i+1:04d}' for i in range(self.max_participants)]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(new_p_list)
        
        logger.info(f'made new partitipant lookup list of {self.max_participants} with seed: {self.seed}')
        
        
        # make a new set of participant numbers:
        # set_trace(port=4444, term_size=(140, 50))
        sessions = list(df[self.params.session].unique())
        
        type_and_unit = df[[self.params.type, self.params.unit]].drop_duplicates().values.tolist()

        total_synthesis = len(participants) * len(sessions) * len (type_and_unit)
        print(f'We will synthesize {total_synthesis} entries of data\nEach entry may contain numerous time points')

        do_time = False
        # this controls what we end up selecting + copy/pasting:
        # we might need to think about the order of things...
        cols_to_select = []
        cols_to_select.append(self.params.participantid)
        cols_to_select.append(self.params.session)
        cols_to_select.append(self.params.type)
        cols_to_select.append(self.params.data)
        cols_to_select.append(self.params.unit)
        if self.params.cols_to_keep is not None:
            if isinstance(self.params.cols_to_keep, str):
                cols_to_select.append(self.params.cols_to_keep)
            elif isinstance(self.params.cols_to_keep, list):
                for c in self.params.cols_to_keep:
                    cols_to_select.append(c)

        if self.params.time_unit is not None and self.params.time_value is not None:
            cols_to_select.append(self.params.time_unit)
            cols_to_select.append(self.params.time_value)
            do_time = True

        count=0
        pbar = tqdm(total=total_synthesis)


        # we'd have to use a different logic if there is no time data...
            # let's do a bunch of queries;

        if do_time:
        
            for i, p in enumerate(participants):
    
                # we use the same seed so each participant will have same type of randomization
                old_participant = p
                my_number = int(re.search(r'\d+', old_participant).group())
                new_participant_number = new_p_list[my_number-1]
                
                for j, s in enumerate(sessions):
                    for k, (t, u) in enumerate(type_and_unit):
    
                        count+=1 # this will likely not do anything.
                        pbar.update(1)
    
                        
                        mask = (
                            (df[self.params.participantid] == p)
                            & (df[self.params.type] == t)
                            & (df[self.params.unit] == u)
                        )
                        
                        if pd.isna(s):
                            mask &= df[self.params.session].isna()
                        else:
                            mask &= df[self.params.session] == s
                        
                        df_to_process = df.loc[mask, cols_to_select]
                        df_to_process[self.params.participantid] = new_participant_number

                        v = np.array(df_to_process[self.params.data])
                        my_msg = f"{p}, sess {s}, t/u {t}/{u}, has {len(v)}  points with {len(set(v))} uniques"

                        try:
                            # make a new v with phase randomization - our site will use seed=1
                            new_v = phase_scramble(v, seed=self.seed)
                            # let's make sure we only use values that occur in the original data                         
                            new_v = map_to_original(v, new_v)
                            # put it back:
                            df_to_process[self.params.data] = new_v

                            # then put that into the big dataframe:
                            self.output.loc[df_to_process.index, df_to_process.columns] = df_to_process
                       
                            logger.info(my_msg)
                            self.history += my_msg + '\n'
                            # set_trace(port=4444, term_size=(140, 50)); 
                        except Exception as e:
                            # e = getattr(e, 'message', repr(e))
                            # set_trace(port=4444, term_size=(140, 50))
                            logger.error(e)
                            logger.error(f'failed: {my_msg}\nerror: {e}')
                            self.errorhistory += my_msg + '\n'

        # we are dealing with singular time points?
        # long-format does allow us to gather ALL of the data, and then do something more smart with it, keeping all of important variability...
        else:
    
            # let's do the participants first then:
            new_participant_list = []
            for i, p in enumerate(participants):
    
                # we use the same seed so each participant will have same type of randomization
                old_participant = p
                my_number = int(re.search(r'\d+', old_participant).group())
                new_participant_number = new_p_list[my_number-1]
                new_participant_list.append(new_participant_number)

            my_mat = []
            my_col_names = []
            my_text_mat = []
            
            for j, s in enumerate(sessions):
                for k, (t, u) in enumerate(type_and_unit):

                    # let's get the info over all the participants:
                    mask = ((df[self.params.type] == t)
                            & (df[self.params.unit] == u)
                        )
                        
                    if pd.isna(s):
                        mask &= df[self.params.session].isna()
                    else:
                        mask &= df[self.params.session] == s

                    df_original = df.loc[mask, cols_to_select].copy()
                    
                    # Only look at the actual data column
                    original_values = df_original[self.params.data].tolist()
                    
                    # Convert values that are recognisably numeric
                    numeric_values = pd.to_numeric(
                        df_original[self.params.data],
                        errors='coerce'
                    )
                    
                    # inf / -inf -> NaN
                    numeric_values = numeric_values.replace([np.inf, -np.inf], np.nan)
                    
                    # Anything that was non-NaN originally but failed numeric conversion = text
                    text_mask = (
                        df_original[self.params.data].notna()
                        & numeric_values.isna()
                    )
                    
                    # Numeric and text vectors
                    numeric_array = numeric_values.to_numpy(dtype=float)
                    
                    text_array = np.full(len(original_values), np.nan, dtype=object)
                    text_array[text_mask.to_numpy()] = np.array(
                        original_values,
                        dtype=object
                    )[text_mask.to_numpy()]
                    
                    
                    my_mat.append(numeric_array.tolist())
                    my_text_mat.append(text_array.tolist())
                    
                    my_col_names.append(f'{s}{t}{u}')

                    
                    # df_to_process = df.loc[mask, cols_to_select].copy()
                    
                    # # 1. Anything that isn't recognisably numerical -> NaN
                    # df_to_process = df_to_process.apply(pd.to_numeric, errors='coerce')
                    
                    # # 2. Explicitly handle inf/-inf as NaN too
                    # df_to_process = df_to_process.replace([np.inf, -np.inf], np.nan)
                    
                    # # 3. Rows where EVERYTHING is 0 -> NaN
                    # all_zero = (df_to_process == 0).all(axis=1)
                    # df_to_process.loc[all_zero, :] = np.nan
                    
                    # # 4. Rows where EVERYTHING is 1 -> NaN
                    # all_one = (df_to_process == 1).all(axis=1)
                    # df_to_process.loc[all_one, :] = np.nan
                    
                    # df_to_process[self.params.participantid] = new_participant_number
                    # now we should have all participants in the df_to_process. Let's run again & check:
                    # my_mat.append(df_to_process['value'].tolist())

                    
                    
                    # set_trace(port=4444, term_size=(140, 50))
                    # my_col_names.append(f'{s}{t}{u}')

            # set_trace(port=4444, term_size=(140, 50))
            
            # 1. Transpose input to subject-major orientation
            m2 = np.array(my_mat).T
            
            # 2. Identify and filter subjects with excessive missing values
            try:
                missing_fraction = np.isnan(m2).mean(axis=1)
            except:
                set_trace(port=4444, term_size=(140, 75))
                
            keep_subject = missing_fraction <= 0.25
            m3 = m2[keep_subject, :]
            
            # 3. Filter features with no variance - or featued that are NAN:

            # 1. Original features that are completely NaN
            all_nan = np.all(np.isnan(m3), axis=0)


            # !! SOME COLUMNS CONTAIN ONLY NANs - FOR SOME REASON
            # WE WILL HANDLE THIS.
            # 2. Calculate variance only for features with at least one value
            zero_var = np.zeros(m3.shape[1], dtype=bool)
            zero_var[~all_nan] = np.nanvar(m3[:, ~all_nan], axis=0) < 1e-10

            # 2. Original features that are completely NaN
            all_nan = np.all(np.isnan(m3), axis=0)
            
            # 3. Features that are actually usable
            meaningful = ~(zero_var | all_nan)
            
            # 4. Keep only usable features for PCA
            X_meaningful_var = m3[:, meaningful]
            
            
            # 4. Impute missing values for clean subjects
            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X_meaningful_var)
            
            # 5. Scale features (mean=0, std=1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # 6. Fit PCA and determine components cutoff
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            explained = pca.explained_variance_ratio_
            cumulative = np.cumsum(explained)
            
            # Number of components needed to reach 75% variance cutoff
            my_ind = np.where(cumulative >= self.PCAthresh)[0][0] + 1
            
            # 7. Low-rank reconstruction for KEPT subjects
            X_pca_k = X_pca[:, :my_ind]
            components_k = pca.components_[:my_ind, :]  # Shape: (k, n_features)
            
            X_reconstructed_scaled = np.dot(X_pca_k, components_k)
            residuals_scaled = X_scaled - X_reconstructed_scaled
            
            # 8. Calculate residual variance and generate synthetic noise
            var_by_var_variance = np.var(residuals_scaled, axis=0)
            std_devs = np.sqrt(var_by_var_variance)
            
            rng = np.random.default_rng(seed=self.seed)
            n_samples, n_features = X_scaled.shape
            simulated_residuals = rng.normal(loc=0.0, scale=std_devs, size=(n_samples, n_features))
            
            # Synthesize kept subjects in original scale
            X_synth_scaled = X_reconstructed_scaled + simulated_residuals
            X_synth_meaningful = scaler.inverse_transform(X_synth_scaled)
            
            # --- ESTIMATE & SYNTHESIZE DROPPED SUBJECTS ---
            
            m2_dropped = m2[~keep_subject, :]
            m3_dropped = m2_dropped[:, meaningful]
            
            X_pca_dropped = np.zeros((m3_dropped.shape[0], my_ind))
            
            for i in range(m3_dropped.shape[0]):
                row = m3_dropped[i, :]
                valid_mask = ~np.isnan(row)
            
                if not np.any(valid_mask):
                    continue

                try:
                # Scale observed features using scaler parameters fitted on clean subjects
                    row_scaled = (row[valid_mask] - scaler.mean_[valid_mask]) / scaler.scale_[valid_mask]
                except:
                    set_trace(port=4444, term_size=(140, 50))
            
                # Target features (already centered by StandardScaler)
                y_observed = row_scaled
            
                # Design matrix: subset component load vectors for valid features
                A_observed = components_k[:, valid_mask].T
            
                # Least squares solver to project available features onto PCA space
                z_i, _, _, _ = np.linalg.lstsq(A_observed, y_observed, rcond=None)
                X_pca_dropped[i, :] = z_i
            
            # Reconstruct dropped subjects in scaled space
            X_reconstructed_dropped_scaled = np.dot(X_pca_dropped, components_k)
            
            # Generate synthetic residuals for dropped subjects
            simulated_residuals_dropped = rng.normal(
                loc=0.0, scale=std_devs, size=X_reconstructed_dropped_scaled.shape
            )
            
            X_synth_dropped_scaled = X_reconstructed_dropped_scaled + simulated_residuals_dropped
            X_synth_dropped_meaningful = scaler.inverse_transform(X_synth_dropped_scaled)
            
            # --- RE-INTEGRATE ALL SUBJECTS & RESTORE FULL MATRIX SHAPE ---
            
            # Combine kept and dropped subjects back into original row order
            X_synth_meaningful_all = np.zeros((m2.shape[0], n_features))
            X_synth_meaningful_all[keep_subject, :] = X_synth_meaningful
            X_synth_meaningful_all[~keep_subject, :] = X_synth_dropped_meaningful  # Fixed variable name
            
            # Re-insert features into original positions
            X_synth_m2 = np.full((m2.shape[0], m2.shape[1]), np.nan, dtype=object)
            
            # Synthesized meaningful features
            X_synth_m2[:, meaningful] = X_synth_meaningful_all
            
            # Restore zero-variance features
            X_synth_m2[:, zero_var] = np.nanmean(m2[:, zero_var], axis=0)


            # restore all of the text stuff:
            # X_synth_m2[np.

            my_text = np.asarray(my_text_mat, dtype=object).T
            mask = ~pd.isna(my_text)
            X_synth_m2[mask] = my_text[mask]
            
            my_mat_synthesized = X_synth_m2.T

          
            # set_trace(port=4444, term_size=(140, 50))

            # GREAT - now the we DO HAVE all of the new values, PUT THEM IN:
            # I am sure this is not the OPTIMAL OPTIMAL way, but best to put back
            # in the same way as I got em. This allows me to not think about it too much
            # while maintaining some sense of 'this is correct'
            
            for j, s in enumerate(sessions):
                for k, (t, u) in enumerate(type_and_unit):

                    my_msg = f"Single point entires across {len(participants)} participants, sess {s}, t/u {t}/{u}"

                    try:
                        count+=1 # this will likely not do anything.
                        pbar.update(len(participants))
    
                        # let's get the info over all the participants:
                        mask = ((df[self.params.type] == t)
                                & (df[self.params.unit] == u)
                            )
                            
                        if pd.isna(s):
                            mask &= df[self.params.session].isna()
                        else:
                            mask &= df[self.params.session] == s
                        
                        df_to_process = df.loc[mask, cols_to_select]
                        # df_to_process[self.params.participantid] = new_participant_number
                        # now we should have all participants in the df_to_process. Let's run again & check:
                        my_mat.append(df_to_process['value'].tolist())
                        # set_trace(port=4444, term_size=(140, 50))
                        which_col = my_col_names.index(f'{s}{t}{u}')
                        values_to_paste = X_synth_m2[:, which_col]
    
                        #  set_trace(port=4444, term_size=(140, 50))
    
    
                        # OK, replace the values:
                        df_to_process[self.params.data] = values_to_paste
                        df_to_process[self.params.participantid] = new_participant_list
    
                        self.output.loc[df_to_process.index, df_to_process.columns] = df_to_process
                        
                    
                        logger.info(my_msg)
                        self.history += my_msg + '\n'
                        # set_trace(port=4444, term_size=(140, 50)); 
                    except Exception as e:
                        e = getattr(e, 'message', repr(e))
                        set_trace(port=4444, term_size=(140, 50))
                        logger.error(e)
                        logger.error(f'failed: {my_msg}\nerror: {e}')
                        self.errorhistory += my_msg + '\n'

            

                        
                    # we likely will have only a single data point
                    # if so, at this stage, we should combine probably the participant x the condition (that we have); analyze the matrix of values somehow
                    # and use a distribution to 'pick values' from that in order to synthesize. BUT - we do need too maintain inter-subject variablility over sessions.
                    # so, what do we do?
                    # ALSO; there's participant - condition - variable type. Right?
                    
                            
    

    def export(self, outfile=None):
        """Export transformed data to CSV."""
    
        if outfile is None:
            # Default: put generated file in synthesized/
            outdir = Path.cwd() / "synthesized"
            outdir.mkdir(parents=True, exist_ok=True)
    
            outfile = outdir / f"{Path(self.filename).stem}_synthesized.csv"
    
        else:
            # User supplied the output path
            outfile = Path(outfile)
    
        self.output.to_csv(outfile, index=False)
    
        msg = f"Written: {outfile}"
        logger.info(msg)
        self.history += msg + "\n"
        print(msg)

        