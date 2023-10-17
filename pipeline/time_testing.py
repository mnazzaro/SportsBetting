import timeit

urls = {
    'Raphael Assuncao': 'http://ufcstats.com/fighter-details/2f13e4020cea5b38',
    'Johnny Eduardo': 'http://ufcstats.com/fighter-details/528c24b99fbaab79'
}
name = 'Raphael Assuncao'

def _find_fighter_in_keys(name: str, urls: dict) -> str:
    for key, val in urls.items():
        if name in key:
            return val

# Timing the function
def wrapper_func():
    _find_fighter_in_keys(name, urls)

time_for_function = timeit.timeit(wrapper_func, number=10000)

# Timing the direct access
def wrapper_dict_access():
    try:
        _ = urls[name]
    except:
        ...

time_for_direct_access = timeit.timeit(wrapper_dict_access, number=10000)

print(f"Time for _find_fighter_in_keys function: {time_for_function}")
print(f"Time for direct dictionary access: {time_for_direct_access}")
