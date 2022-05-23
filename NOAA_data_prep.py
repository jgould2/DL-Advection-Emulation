import netCDF4 as nc
import numpy as np
import tensorflow as tf

fn = '/home/jacob/PycharmProjects/DL Advection Emulation/NOAA Data/' + 'inp.20190806.120300.fv_core.res.tile1.nc'
#fn = '/home/jacob/PycharmProjects/DL Advection Emulation/NOAA Data/' + 'inp.20190806.120300.fv_tracer.res.tile1.nc'
ds = nc.Dataset(fn)
#print(ds)
#print(type(ds['u'][:]))
#print(ds['u'][:])
#print(ds.variables.keys())
patch = [64, 64]
grid = [232, 396]
tracers = ['sphum', 'liq_wat', 'rainwat', 'ice_wat', 'snowwat', 'graupel',
 'o3mr', 'no2', 'no', 'o3', 'no3', 'h2o2', 'n2o5', 'hno3', 'hono', 'pna', 'so2', 'sulf', 'pan', 'pacd', 'aacd',
 'ald2', 'panx', 'form', 'mepx', 'meoh', 'rooh', 'ntr1', 'ntr2', 'facd', 'co', 'aldx', 'glyd', 'gly', 'mgly',
 'etha', 'etoh', 'ket', 'par', 'acet', 'prpa', 'ethy', 'eth', 'ole', 'iole', 'isop', 'ispd', 'intr', 'ispx',
 'hpld', 'opo3', 'epox', 'terp', 'benzene', 'cres', 'open', 'tol', 'xopn', 'xylmn', 'naph', 'cat1', 'cron',
 'opan', 'ech4', 'cl2', 'hocl', 'fmcl', 'hcl', 'clno2', 'sesq', 'soaalk', 'vlvpo1', 'vsvpo1', 'vsvpo2', 'vsvpo3',
 'vivpo1', 'vlvoo1', 'vlvoo2', 'vsvoo1', 'vsvoo2', 'vsvoo3', 'pcvoc', 'form_primary', 'ald2_primary', 'butadiene13',
 'acrolein', 'acro_primary', 'tolu', 'hg', 'hgiigas', 'aso4j', 'aso4i', 'anh4j', 'anh4i', 'ano3j', 'ano3i',
 'aalk1j', 'aalk2j', 'axyl1j', 'axyl2j', 'axyl3j', 'atol1j', 'atol2j', 'atol3j', 'abnz1j', 'abnz2j', 'abnz3j',
 'apah1j', 'apah2j', 'apah3j', 'atrp1j', 'atrp2j', 'aiso1j', 'aiso2j', 'asqtj', 'aorgcj', 'aecj', 'aeci', 'aothrj',
 'aothri', 'afej', 'aalj', 'asij', 'atij', 'acaj', 'amgj', 'akj', 'amnj', 'acors', 'asoil', 'numatkn', 'numacc',
 'numcor', 'srfatkn', 'srfacc', 'srfcor', 'ah2oj', 'ah2oi', 'ah3opj', 'ah3opi', 'anaj', 'anai', 'aclj', 'acli',
 'aseacat', 'aclk', 'aso4k', 'anh4k', 'ano3k', 'ah2ok', 'ah3opk', 'aiso3j', 'aolgaj', 'aolgbj', 'aglyj', 'apcsoj',
 'alvpo1i', 'asvpo1i', 'asvpo2i', 'alvpo1j', 'asvpo1j', 'asvpo2j', 'asvpo3j', 'aivpo1j', 'alvoo1i', 'alvoo2i',
 'asvoo1i', 'asvoo2i', 'alvoo1j', 'alvoo2j', 'asvoo1j', 'asvoo2j', 'asvoo3j', 'nh3', 'sv_alk1', 'sv_alk2', 'sv_xyl1',
 'sv_xyl2', 'sv_tol1', 'sv_tol2', 'sv_bnz1', 'sv_bnz2', 'sv_pah1', 'sv_pah2', 'sv_trp1', 'sv_trp2', 'sv_iso1',
 'sv_iso2', 'sv_sqt', 'lv_pcsog', 'pm25at', 'pm25ac', 'pm25co', 'cld_amt']

exit()
times = ['120300', '120600', '120900', '121200', '121500', '121800']
test_time = times[0]
zs = [59, 60, 61, 62, 63]
print(len(tracers))
count = 0
nan_count = 0
train_indx = []
val_indx = []
test_indx = []
max_max_tracer = 0
max_max_info = []
nan_vals = []
for time in times:
    tracers_in = nc.Dataset('/home/jacob/PycharmProjects/DL Advection Emulation/NOAA Data/inp.20190806.' + time + '.fv_tracer.res.tile1.nc')
    tracers_out = nc.Dataset('/home/jacob/PycharmProjects/DL Advection Emulation/NOAA Data/out.20190806.' + time + '.fv_tracer.res.tile1.nc')
    phys = nc.Dataset('/home/jacob/PycharmProjects/DL Advection Emulation/NOAA Data/inp.20190806.' + time + '.fv_core.res.tile1.nc')
    print(str(time) + ': winds')
    for z in zs:
        slice = 0
        u = np.array(phys['u'][0][z])
        v = np.array(phys['v'][0][z])
        mean = (np.mean(u) + np.mean(v))/2
        std = np.sqrt((np.sum(np.square(u - mean)) + np.sum(np.square(v - mean)))/(u.size +v.size))
        max = np.max([np.max(np.abs(u)), np.max(np.abs(v))])
        print(mean, std, max)
        #np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Winds/stats_' + time + '_' + str(z) + '.npy', np.array([mean, std]))
        for x_slice in range(int(grid[0] / patch[0])):
            for y_slice in range(int(grid[1] / patch[1])):
                u_p = phys['u'][0][z][x_slice * patch[0]: (1 + x_slice) * patch[0] + 1, y_slice * patch[1]: (1 + y_slice) * patch[1]]
                v_p = phys['v'][0][z][x_slice * patch[0]: (1 + x_slice) * patch[0], y_slice * patch[1]: (1 + y_slice) * patch[1] + 1]
                if max != 0:
                    u_scale = u_p/max
                    v_scale = v_p/max
                else:
                    u_scale = 0
                    v_scale = 0
                if std != 0:
                    u_norm = (u_p - mean)/std
                    v_norm = (v_p - mean)/std
                else :
                    u_norm = u_p - mean
                    v_norm = v_p - mean
                np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Winds/u_normed_' + time + '_' + str(z) + '_' + str(slice) + '.npy', u_norm.data)
                np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Winds/v_normed_' + time + '_' + str(z) + '_' + str(slice) + '.npy', v_norm.data)
                np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Winds/u_scaled_' + time + '_' + str(z) + '_' + str(slice) + '.npy', u_scale.data)
                np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Winds/v_scaled_' + time + '_' + str(z) + '_' + str(slice) + '.npy', v_scale.data)
                slice += 1
    print(str(time) + ': tracers')
    for tracer in tracers:
        for z in zs:
            np.random.seed(count)
            val_slice = np.random.randint(0, int(grid[0]/patch[0]) * int(grid[1]/patch[1]), 2)
            slice = 0
            if z == zs[-1]:
                tracer_max = np.max(tracers_in[tracer][0][z-1:])
            elif z == zs[0]:
                tracer_max = np.max(tracers_in[tracer][0][z:z+2])
            else:
                tracer_max = np.max(tracers_in[tracer][0][z-1:z+2])
            for x_slice in range(int(grid[0]/patch[0])):
                for y_slice in range(int(grid[1]/patch[1])):
                    tracer_in = np.expand_dims(np.array(tracers_in[tracer][0][z][x_slice * patch[0] : (1 + x_slice) * patch[0], y_slice * patch[1] : (1 + y_slice) * patch[1]]), axis=-1)
                    tracer_out = np.expand_dims(np.array(tracers_out[tracer][0][z][x_slice * patch[0] : (1 + x_slice) * patch[0], y_slice * patch[1] : (1 + y_slice) * patch[1]]), axis=-1)
                    tracer_concat = np.concatenate([tracer_in, tracer_out], axis=-1)
                    if tracer_max != 0:
                        tracer_concat /= tracer_max
                    t_max = np.max(tracer_concat)
                    #if max_max_tracer < t_max:
                        #max_max_info = [time, str(z), str(slice), tracer, str(tracer_max)]
                        #max_max_tracer = np.max([max_max_tracer, t_max])
                    if t_max > 5:
                        nan_count += 1
                        nan_vals.append(t_max)
                    else:
                        if time == test_time:
                            test_indx.append(count)
                        elif slice == val_slice[0] or slice == val_slice[1]:
                            val_indx.append(count)
                        else:
                            train_indx.append(count)
                    #print(nan_count, tracer_max, t_max, np.max(tracer_out), np.max(tracer_in), max_max_tracer, max_max_info)
                    print(nan_count, count, nan_count/(count+1), count/104760)
                    #print('**********************')
                    np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/tracer_'+str(count)+'.npy', tracer_concat)
                    np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/tracer_'+str(count)+'_info.npy', np.array([time, str(z), str(slice), tracer, tracer_max]))
                    count += 1
                    slice += 1
print(np.mean(nan_vals), np.std(nan_vals), np.min(nan_vals), np.max(nan_vals))
np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/test_indx.npy', test_indx)
np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/val_indx.npy', val_indx)
np.save('/home/jacob/PycharmProjects/DL Advection Emulation/Model Applications for NOAA/Prepped Data/Tracers/train_indx.npy', train_indx)
print(len(test_indx), len(val_indx), len(train_indx))
print('done')
