import asyncio
import threading
import time

from bleak import BleakScanner, BleakClient


class Bittle:
    def __init__(self):
        self.bittle = discover_devices()
        self.bluetooth, self.bluetooth_thread = connect(self.bittle.address)

        print('Connected to Bittle!')

        time.sleep(10)  # Give Bittle time to boot up

        self.reader, self.writer = [service.characteristics for service in self.bluetooth.services][0]

        reading = self.bluetooth.start_notify(self.reader, self.reading)  # Asynchronous reading Bittle measurements
        self.reading_thread = parallelize(reading)  # Begin reading Bittle measurements in real-time

        self.action_done = True

    def reading(self, _, data: bytearray):
        data = data.decode('ISO-8859-1')
        if 'i' in data:
            self.action_done = True

    def step(self, action):
        async def act():
            self.action_done = False
            await self.bluetooth.write_gatt_char(self.writer, action)  # Triggers a reaction

        while not self.action_done:
            pass

        asyncio.run(act())

    def disconnect(self):
        parallelize(self.bluetooth.disconnect()).set()
        self.bluetooth_thread.set()
        self.reading_thread.set()

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

    event = parallelize(_connect())

    return bluetooth, event


def parallelize(run):
    event = asyncio.Event()
    event.clear()

    def _parallelize():
        async def _run():
            try:
                await run
            except RuntimeError as e:
                if 'got Future <Future pending> attached to a different loop' not in str(e):
                    raise e
            await event.wait()  # Keep running

        asyncio.run(_run())

    thread = threading.Thread(target=_parallelize)
    thread.start()

    return event


commands = map(encode, [[45, 0, 0, 0, 0, 0, 0, 0, 25, -45, -45, 35, 66, 36, 36, 36],
                        [-45, 0, 0, 0, 0, 0, 0, 0, 45, 45, -45, 48, 36, 36, 36, 36],
                        [45, 0, 0, 0, 0, 0, 0, 0, 45, -45, 45, 45, 36, 36, 36, 36],
                        [-45, 0, 0, 0, 0, 0, 0, 0, 45, 45, 45, 45, 36, 36, -36, 36]])

bittle = Bittle()
for command in commands:
    bittle.step(command)
