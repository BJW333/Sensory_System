import threading
import time
import requests
import platform
from context_fusion import CONTEXT

def get_location_ip():
    try:
        resp = requests.get("http://ip-api.com/json/")
        data = resp.json()
        city = data.get('city')
        region = data.get('regionName')
        country = data.get('country')
        lat = data.get('lat')
        lon = data.get('lon')
        return {'city': city, 'region': region, 'country': country, 'lat': lat, 'lon': lon}
    except Exception as e:
        print("[LocationMonitor-IP] Error:", e)
        return None

def get_location_geocoder():
    try:
        import geocoder
        g = geocoder.ip('me')
        if g.ok and g.latlng:
            lat, lon = g.latlng
            city = g.city
            country = g.country
            return {'city': city, 'region': '', 'country': country, 'lat': lat, 'lon': lon}
    except Exception as e:
        print("[LocationMonitor-Geocoder] Error:", e)
    return None

def get_location_macos():
    try:
        import objc
        from Cocoa import NSObject
        from CoreLocation import CLLocationManager
        from PyObjCTools import AppHelper

        class Delegate(NSObject):
            def init(self):
                self = objc.super(Delegate, self).init()
                self.location = None
                return self

            def locationManager_didUpdateLocations_(self, manager, locations):
                loc = locations[-1]
                self.location = {
                    'lat': loc.coordinate().latitude,
                    'lon': loc.coordinate().longitude
                }

        delegate = Delegate.alloc().init()
        manager = CLLocationManager.alloc().init()
        manager.setDelegate_(delegate)
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
        print("[LocationMonitor-macOS] Waiting for location permission and fix...")
        timeout = 15
        while delegate.location is None and timeout > 0:
            time.sleep(1)
            timeout -= 1
        manager.stopUpdatingLocation()
        return delegate.location
    except Exception as e:
        print("[LocationMonitor-macOS] Error:", e)
        return None

def get_best_location():
    if platform.system() == "Darwin":
        loc = get_location_macos()
        if loc: return loc
    loc = get_location_geocoder()
    if loc: return loc
    return get_location_ip()

def get_weather(lat, lon, api_key):
    if not api_key:
        print("[WeatherMonitor] No API key provided for weather.")
        return None
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        resp = requests.get(url)
        data = resp.json()
        summary = data['weather'][0]['description']
        temp = data['main']['temp']
        return {'summary': summary, 'temp': temp}
    except Exception as e:
        print("[WeatherMonitor] Error:", e)
        return None

def monitor_location_and_weather(api_key, interval=900):
    last_loc = None
    last_weather = None
    while True:
        loc = get_best_location()
        if loc:
            if loc != last_loc:
                CONTEXT.update("user_location", loc)
                last_loc = loc
            weather = get_weather(loc['lat'], loc['lon'], api_key)
            if weather and weather != last_weather:
                CONTEXT.update("weather", weather)
                last_weather = weather
        time.sleep(interval)

def start_location_weather_monitor_thread(api_key, interval=900):
    t = threading.Thread(target=monitor_location_and_weather, args=(api_key, interval), daemon=True)
    t.start()