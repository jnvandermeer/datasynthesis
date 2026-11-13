## DATA SYNTHESIS ON WIDE TABLES WITH SYNTHETIC DATA VAULT (SDV)
#
# Generate synthetic data from a wide table using SDV (Synthetic Data Vault).


from importlib.metadata import metadata
import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer



class DataSynthesisWideSDV:

    # Constructor
    def __init__(self):
        self.dataframe = None
        self.metadata = None
        self.params = SimpleNamespace(
            participantid=None,
            session=None,
            trial=None,
            cols_to_drop=None,                      
        )

    # Load wide table and generate primary key and metadata table
    # Automatically assumes participantid + session + trial (if present) is the primary key
    def load_wide(self, filename, participantid, session, trial=None, cols_to_drop=None):

        # Update parameters
        self.params.participantid = participantid
        self.params.session = session
        self.params.trial = trial
        self.params.cols_to_drop = cols_to_drop
        
        # Load the wide table
        self.dataframe = pd.read_csv(filename)

        # Drop any columns if required
        if cols_to_drop:
            self.dataframe = self.dataframe.drop(columns=cols_to_drop)


        # Recode trials to monotonically increasing integers if trial column is specified
        # This should ideally be done in synthesize(), but it is easiest to do it when 
        # the table is in wide form as we can use groupby() and cumcount()
        if trial is not None:
            self.dataframe[trial] = self.dataframe.groupby([participantid, session]).cumcount() + 1

        # Get the participant-session-trial metrics
        self.nparticipants = self.dataframe[participantid].nunique()
        self.nsessions = self.dataframe[session].nunique()
        self.ntrials = self.dataframe[trial].nunique() if trial is not None else 1


        # SDV can"t handle deep nested designs in a single table, e.g. participant-session-trial,
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
        # (hardcoded max 999 trials per session, but this is unlikely to be exceeded in practice)
        if trial is None:
            self.dataframe["sequence"] = self.dataframe[session]*1000
        else:
            self.dataframe["sequence"] = ((self.dataframe[session] - 1) * 1000) + self.dataframe[trial]

        # Order the columns by participantid and sequence just in case they may be out of order
        self.dataframe = self.dataframe.sort_values(by=[participantid, "sequence"]).reset_index(drop=True)


        # Generate table metadata
        self.metadata = Metadata.detect_from_dataframes(data={"data": self.dataframe})
        self.metadata.update_column(column_name=participantid, sdtype="id")
        self.metadata.update_column(column_name="sequence", sdtype="numerical")
        self.metadata.update_column(column_name=session, sdtype="categorical")  
        self.metadata.update_column(column_name=trial, sdtype="categorical")        
        self.metadata.set_sequence_key(participantid)
        self.metadata.set_sequence_index("sequence")



    # Learn the structure of the data
    def learn(self):
        self.synthesizer = PARSynthesizer(metadata=self.metadata, verbose=True)  # Use PARSynthesizer for sequential data, display progress bar
        self.synthesizer.fit(data={"data": self.dataframe})     # Takes a dict, which is the same at what is in the metadata dict



    # Generate synthetic data using SDV
    # Specify new table size (participants, session, trials) if required,
    # otherwise use the same size as the original table
    def synthesize(self, nparticipants=None, nsessions=None, ntrials=None):

        # Size of synthetic output data
        # Default to input table unless specified differently
        np = nparticipants if nparticipants is not None else self.nparticipants
        ns = nsessions if nsessions is not None else self.nsessions
        nt = ntrials if ntrials is not None else self.ntrials

        # Actual synthetic table size
        nseqs = np
        nslen = ns * nt
        
        # Generate synthetic data (returns dict, need to unwrap)
        synth_dict = self.synthesizer.sample_sequences(num_sequences=nseqs, sequence_length=nslen)
        self.synthetic = synth_dict["data"]

        # Reconstruct the session and trial columns from the sequence column
        self.synthetic[self.params.session] = (self.synthetic["sequence"] - 1) // 1000 + 1
        self.synthetic[self.params.trial] = (self.synthetic["sequence"] - 1) % 1000 + 1
        self.synthetic = self.synthetic.drop(columns="sequence")