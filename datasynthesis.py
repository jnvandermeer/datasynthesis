from pathlib import Path
import pandas as pd
from types import SimpleNamespace
from pudb.remote import set_trace
import numpy as np
from scramblers import phase_scramble
import logging
from tqdm import tqdm
import re



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

    def __init__(self, seed=1, max_participants=1000):
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
        if do_time:

            # let's do a bunch of queries;
            for i, p in enumerate(participants):
                for j, s in enumerate(sessions):
                    for k, (t, u) in enumerate(type_and_unit):

                        count+=1 # this will likely not do anything.
                        pbar.update(1)
    
                        # my_conditions = []
                        # my_conditions.append((self.params.participantid, "==",p))

                        # if pd.isna(s):
                        #     # handle missing session separately
                        #     my_conditions.append((self.params.session, "is", None))
                        # else:
                        #     my_conditions.append((self.params.session, "==", s))    
                        
                        # # my_conditions.append((self.params.session, "==", s))
                        # my_conditions.append((self.params.type, "==", t))
                        # my_conditions.append((self.params.unit, "==", u))
    
                        # query = " and ".join(
                        #     f"`{col}` {op} {value!r}"
                        #     for col, op, value in my_conditions
                        # )                    

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

                        
                        
                        # select the data - make smaller dataframe, then grab values
                        # df_to_process = df.query(query)[cols_to_select]
                        v = np.array(df_to_process[self.params.data])

                        my_msg = f"{p}, sess {s}, t/u {t}/{u}, has {len(v)}  points with {len(set(v))} uniques"

                        try:
                            # make a new v with phase randomization - our site will use seed=1
                            new_v = phase_scramble(v, seed=self.seed)
                            
                            # let's make sure we only use values that occur in the original data                         
                            new_v = map_to_original(v, new_v)
        
                            # put it back:
                            df_to_process[self.params.data] = new_v


                            # replace the participant based on our generated list of max_participants;
                            # we use the same seed so each participant will have same type of randomization
                            old_participant = p
                            my_number = int(re.search(r'\d+', old_participant).group())
                            df_to_process[self.params.participantid] = new_p_list[my_number-1]
                            # set_trace(port=4444, term_size=(140, 50))
                            
                            # then put that into the big dataframe:
                            self.output.loc[df_to_process.index, df_to_process.columns] = df_to_process
        
                       
                            logger.info(my_msg)
                            self.history += my_msg + '\n'
                                    
                            # allow me optionally to check what's going on here
                            # print('term size = 140x50')
                            # set_trace(port=4444, term_size=(140, 50)); 
                        except Exception as e:
                            # e = getattr(e, 'message', repr(e))
                            # set_trace(port=4444, term_size=(140, 50))
                            logger.error(e)
                            logger.error(f'failed: {my_msg}\nerror: {e}')
                            self.errorhistory += my_msg + '\n'
    

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

        