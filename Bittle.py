import asyncio
import threading
import time

from bleak import BleakScanner, BleakClient

import numpy as np


class Bittle:
    """
    A general-purpose environment:

    Must accept: **kwargs as init arg.

    Must have:

    (1) a "step" function, action -> exp
    (2) "reset" function, -> exp
    (3) "render" function, -> image
    (4) "episode_done" attribute
    (5) "obs_spec" attribute which includes:
        - "shape", "mean", "stddev", "low", "high" (the last 4 can be None)
    (6) "action-spec" attribute which includes:
        - "shape", "discrete_bins" (should be None if not discrete), "low", "high", and "discrete"
    (7) "exp" attribute containing the latest exp

    Recommended: Discrete environments should have a conversion strategy for adapting continuous actions (e.g. argmax)

    An "exp" (experience) is an AttrDict consisting of "obs", "action" (prior to adapting), "reward", and "label"
    as numpy arrays with batch dim or None. "reward" is an exception: should be numpy array, can be empty/scalar/batch.

    ---

    Can optionally include a frame_stack, action_repeat method.

    """
    def __init__(self, **kwargs):
        self.episode_done = False

        print('Discovering devices...')

        self.bittle = discover_devices()
        self.bluetooth, self.bluetooth_thread = connect(self.bittle.address)

        print('Connected to Bittle!')

        time.sleep(10)  # Give Bittle time to boot up

        self.reader, self.writer = [service.characteristics for service in self.bluetooth.services][0]

        parallelize(self.bluetooth.start_notify(self.reader, self.reading))  # Asynchronous reading Bittle measurements

        self.action_done = True

        self.obs_spec = {'shape': (3,),  # Update
                         'mean': None,
                         'stddev': None,
                         'low': None,
                         'high': None}

        self.action_spec = {'shape': (16,),
                            'discrete_bins': None,
                            'low': -90,
                            'high': 90,
                            'discrete': False}

        self.exp = AttrDict()  # Experience dictionary

        # self.frames = deque([], frame_stack or 1)  # TODO

    def reading(self, _, data: bytearray):
        data = data.decode('ISO-8859-1')
        print(data)
        if 'i' in data:
            self.action_done = True
        else:
            try:
                measurement = np.array(list(map(float, data.strip('\r\nv\r\n').split('\t'))))

                if measurement.shape == self.obs_spec['shape']:
                    self.exp.obs = measurement
            except ValueError:
                return

    def step(self, action):
        self.action_done = False
        parallelize(self.bluetooth.write_gatt_char(self.writer, encode(action)))  # Triggers a reaction

        self.exp.obs = None
        self.exp.action = action
        self.exp.reward = np.array([])
        self.exp.label = None

        while not self.action_done:
            pass

        parallelize(self.bluetooth.write_gatt_char(self.writer, b'v'))

        while self.exp.obs is None:
            pass

        print(self.exp)

        return self.exp  # Experience

    def reset(self):
        ...

    def render(self):
        ...

    def disconnect(self):
        while not self.action_done:
            pass

        parallelize(self.bluetooth.disconnect())
        self.bluetooth_thread.set()

        print('Disconnected from Bittle')


servos = range(16)  # 16 degrees of freedom


def encode(rotations: list):
    return ('i ' + ' '.join(map(str, [*sum(zip(servos, rotations), ())]))).encode('utf-8')


def discover_devices():
    async def _discover_devices():
        devices = await BleakScanner.discover(timeout=15)
        for d in devices:
            if d.name and 'Petoi' in d.name:
                return d
        assert False, 'Could not find Petoi device in *disconnected* Bluetooth devices. ' \
                      'Make sure Your Bittle is on and your machine is *not* yet connected to it. ' \
                      'Can only auto-connect disconnected devices. Consider restarting the Bittle or your Bluetooth.'
    return asyncio.run(_discover_devices())


def connect(address):
    bluetooth = BleakClient(address)

    async def _connect():
        await bluetooth.connect()

    event = parallelize(_connect(), forever=True)

    return bluetooth, event


def parallelize(run, forever=False):
    event = asyncio.Event()
    event.clear()

    async def launch():
        try:
            await run
        except RuntimeError as e:
            if 'got Future <Future pending> attached to a different loop' not in str(e):
                raise e
        if forever:
            await event.wait()  # Keep running

    thread = threading.Thread(target=asyncio.run, args=(launch(),))
    thread.start()

    return event


# Access a dict with attribute or key (purely for aesthetic reasons)
class AttrDict(dict):
    def __init__(self, _dict=None):
        super().__init__()
        self.__dict__ = self
        if _dict is not None:
            self.update(_dict)


commands = [np.array(command, dtype='float32') for command in [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                               [-45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                               [45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                               [-45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                               [45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                               [-90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]

bittle = Bittle()
for command in commands:
    bittle.step(command)
bittle.disconnect()
