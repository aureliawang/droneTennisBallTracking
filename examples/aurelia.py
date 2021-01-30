from time import sleep
import tellopy

def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)

def test():
    drone = tellopy.Tello()
    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        drone.connect()
        drone.wait_for_connection(60.0)
        drone.takeoff()
        sleep(5)
        drone.flip_back()
        drone.left(15)
        sleep(5)
        drone.right(15)
        sleep(5)
        drone.flip_forwardright()
        drone.down(50)
        sleep(5)
        drone.land()
        sleep(5)
    
    except Exception as ex:
        print(ex)
    finally:
        drone.quit()

if __name__ == '__main__':
    test()

