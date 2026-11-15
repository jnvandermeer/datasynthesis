## DATA SYNTHESIS ON DISCRETE DATA WITH SYNTHETIC DATA VAULT (SDV)


from importlib.metadata import metadata
import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer
import pickle as pkl



class DataSynthesisDiscreteSDV:

    # Constructor
    def __init__(self):
        self.params = SimpleNamespace(
            filename=None,
            participantid=None,
            session=None,
            trial=None,
            cols_to_drop=None, 
            expected_nsessions=None,
            expected_ntrials=None,
            participant_regex=r"[1-9][0-9]{0,17}"    # default: any integer, up to 17 digits (need to limit size as * in regex can lead to integer overflow errors)                   
        )



    # Load table of discrete data and prepare it for SDV 
    def load(self, filename, participantid, session, expected_nsessions=1, trial=None, expected_ntrials=1, cols_to_drop=None, participant_regex=None):
        """ Load a long-ish table of discrete data and prepare it for synthetic data generation using SDV. 
            Tables should have variables as columns, samples as rows. That is, each participant-session 
            or participant-session-trial combination should be one data row. The order of rows should
            represent the temporal sequence of data. E.g. longitudinal participant-session-trial set ups,
            like HISS, each participant has multiple sessions, each session has many trials. For a given
            session for a participant, trials should be stacked first to last, then blocks of session rows 
            should be stacked earliest to latest. """


        # Load the wide table
        self.dataframe_original = pd.read_csv(filename)

        # Make a deep copy that will be used for SDV
        self.dataframe = self.dataframe_original.copy()

        # If not trial column specified, create a dummy one filled with 1
        if trial is None:
            self.dataframe["trial"] = 1

        # Drop any columns if required
        if cols_to_drop:
            self.dataframe = self.dataframe.drop(columns=cols_to_drop)

        # Report the number of dropped rows
        self.ncols_dropped = self.dataframe_original.shape[1] - self.dataframe.shape[1]
        print("Number of dropped columns: %d" % self.ncols_dropped)


        # Update parameters
        self.params.filename = filename
        self.params.participantid = participantid
        self.params.session = session
        self.params.trial = trial
        self.params.cols_to_drop = cols_to_drop

        # Save the expected participant-session-trial metrics
        self.nparticipants = self.dataframe[participantid].nunique()
        self.expected_sessions = expected_nsessions
        self.expected_ntrials = expected_ntrials



        # Need to deal with the situation where FEWER trials are present than expected. So
        # create a new df with the expected total number of rows and left merge the original
        # df so that missing rows are filled with NaN. Then impute the NaNs.
        #
        # Note: Left merge drops any unmatched right df rows. So if a participant has MORE 
        # trials than expected, ensure the excess or unwanted trials are removed so that
        # required trials are not potentially dropped. 

        # Recode trials to monotonically increasing integers if trial column is specified
        self.dataframe[trial] = self.dataframe.groupby([participantid, session]).cumcount() + 1

        # Build the new df
        participants = self.dataframe[participantid].unique()
        grid = pd.MultiIndex.from_product([participants, range(1, expected_nsessions + 1), range(1, expected_ntrials + 1)], names=[participantid, session, trial]).to_frame(index=False)        

        # Left merge the original df into a new df, missing trials become NaNs
        dataframe_padded = pd.merge(grid, self.dataframe, on=[participantid, session, trial], how='left')

        # Report the number of dropped rows
        # If this is non-zero then you need to manually check if important trials were dropped
        self.nrows_dropped = self.dataframe.shape[0] - dataframe_padded.shape[0]
        print("Number of dropped rows: %d" % self.nrows_dropped)

        # Mean impute NaNs
        # Note: this will overwrite trial codes but they are manually recoded next anyway
        # Also no need for SimpleImputer as it can be done easily with base Python)
        cols_to_impute = dataframe_padded.columns.difference([participantid, session])
        dataframe_padded[cols_to_impute] = dataframe_padded.groupby([participantid, session])[cols_to_impute].transform(lambda x: x.fillna(x.mean()))
        self.dataframe = dataframe_padded.copy()    # overwrite

        # Recode trials as monotonically increasing ids incase imputation overwrote them
        self.dataframe[trial] = self.dataframe.groupby([participantid, session]).cumcount() + 1




        # SDV can't handle deep nested designs in a single table, e.g. participant-session-trial,
        # so we need to recode the trial column to monotonically increasing integers and merge
        # with session to provide a single sequence number per participants. We can then use
        # PARSynthetizer to generate synthetic data for each participant-sequence pair. The
        # synthetic output may not have monotonically increasing sequence numbers, but we can
        # manually recode them back to the original sequence and trial numbers after synthesis.
        #
        # We don't drop the original session and trial columns, but we keep them as categorical
        # so the PARSynthetizer has longituduinal information when learning the structure of the
        # data but we will need to overwrite the synthetic session and trial columns after synthesis
        # as these will not be generated correctly by the PARSynthetizer.
        #
        # Creating a session-trial sequence rather than participant-session sequence means
        # the PARSynthetizer will learn the longitudinal structure of the data across the
        # sessions but it will be less focussed on within session structure. If the within session
        # structure is important, then we can create a participant-session sequence, however,
        # this will require complex recoding, but can be implemented later if required.

        # Create a sequence column that can be recoded back to a session and trial after synthesis
        self.dataframe["sequence"] = self.dataframe.groupby(participantid).cumcount() + 1

        # Order the columns by participantid and sequence just in case they may be out of order
        self.dataframe = self.dataframe.sort_values(by=[participantid, "sequence"]).reset_index(drop=True)


        # Generate table metadata
        self.metadata = Metadata.detect_from_dataframes(data={"data": self.dataframe})
        self.metadata.update_column(column_name=participantid, sdtype="id", regex_format=participant_regex)
        self.metadata.update_column(column_name="sequence", sdtype="numerical")
        self.metadata.update_column(column_name=session, sdtype="categorical")  
        self.metadata.update_column(column_name=trial, sdtype="categorical")        
        self.metadata.set_sequence_key(participantid)
        self.metadata.set_sequence_index("sequence")





    # Learn the structure of the data
    def learn(self, savefile=None):
        """ Learn the structure of the data with SDV PARSynthesizer. """        
        
        self.synthesizer = PARSynthesizer(metadata=self.metadata, verbose=True)  # Use PARSynthesizer for sequential data, display progress bar
        self.synthesizer.fit(data={"data": self.dataframe})     # Takes a dict, which is the same at what is in the metadata dict




    # Generate synthetic data using SDV
    # Specify new table size (participants, session, trials) if required,
    # otherwise use the same size as the original table
    def synthesize(self, nparticipants=None, nsessions=None, ntrials=None):
        """ Synthesise a new data table from the original data. By default the new table will have the same
            participant-session-trial structure unless a new table size is provided. """

        # Size of synthetic output data
        # Default to input table unless specified differently
        np = nparticipants if nparticipants is not None else self.nparticipants
        ns = nsessions if nsessions is not None else self.expected_sessions
        nt = ntrials if ntrials is not None else self.expected_ntrials

        # Actual synthetic table size
        nseqs = np
        nslen = ns * nt
        
        # Generate synthetic data (returns dict, need to unwrap)
        synth_dict = self.synthesizer.sample_sequences(num_sequences=nseqs, sequence_length=nslen)
        self.synthetic = synth_dict["data"]

        # Reconstruct the session and trial columns from the sequence column
        self.synthetic[self.params.session] = (self.synthetic["sequence"] - 1) // nt + 1
        self.synthetic[self.params.trial] = (self.synthetic["sequence"] - 1) % nt + 1
        self.synthetic = self.synthetic.drop(columns="sequence")




    # Export synthetic tables to CSV
    def export(self, outfile=None):
        """ Write synthetic table to CSV. """

        # File name and folder
        if outfile is None:
            outdir, original = os.path.split(self.params.filename)
            filename = original[:-4] + "_synthesized.csv"
            outfile = os.path.join(outdir, filename)
            print(filename)
        else:
            outdir, _ = os.path.split(outfile)

        # Make folder if required
        os.makedirs(outdir, exist_ok=True)

        # Write to CSV
        self.synthetic.to_csv(outfile, index=False)

        print("Synthetic data written to: \"%s\"" % outfile)




    # Save the full data structure
    def save_model(self, outfile, synthesizer_only=False):
        """ Pickle the full data structure or just the synthesizer. Output filename
            should have extension .pkl (not checked). """ 

        # Save the synthesizer only using SDV's built-in save function
        if synthesizer_only:
            self.synthesizer.save(outfile)

        # Otherwise Pickle the full data structure, including original data set,
        # metadata, synthesizer model, and synthetic data set
        else:
            with open(outfile, "wb") as fid:
                pkl.dump(self, fid)



