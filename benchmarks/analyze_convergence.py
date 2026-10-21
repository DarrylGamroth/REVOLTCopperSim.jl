"""Reduce archived Copper CUDA outputs; no GPU or Julia runtime required."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, platform, tomllib
from pathlib import Path
import numpy as np

PROBE_NAMES = ("flat", "piston", "tilt_x_20nm", "tilt_y_20nm", "focus_20nm",
 "astigmatism_20nm", "coma_20nm", "tilt_x_100nm", "focus_100nm",
 "fourier_4_20nm", "fourier_8_20nm", "fourier_12_20nm", "mixed_100nm")
SUPPORT = np.fromfunction(lambda i,j:(i-14.5)**2+(j-14.5)**2 <= 15**2,(30,30))
# Same Julia column-major selection; transpose AOS(x,y) to image(row=y,column=x).
OFFSETS = np.flatnonzero(SUPPORT.ravel(order="F"))
ORIGINS = ((1,1),(33,1),(33,33),(1,33))


def load_record(path):
    data=tomllib.loads(path.read_text())
    arrays={}
    for key in ("rates","interaction"):
        record=data[key]
        content=(path.parent/record["file"]).read_bytes()
        assert hashlib.sha256(content).hexdigest()==record["sha256"]
        arrays[key]=np.frombuffer(gzip.decompress(content),dtype="<f4").reshape(record["shape"],order="F").astype(np.float64)
        assert np.isfinite(arrays[key]).all()
    return data,arrays


def slopes(rates,divisor):
    image=rates.T
    pupils=[image[r:r+30,c:c+30].ravel(order="F")[OFFSETS] for r,c in ORIGINS]
    a,b,c,d=pupils
    return np.concatenate(((a-b+d-c)/divisor,(a-d+b-c)/divisor))


def relative(actual,expected):
    denominator=np.linalg.norm(expected)
    return np.linalg.norm(actual-expected)/denominator if denominator else (0.0 if np.array_equal(actual,expected) else float("inf"))


def compare(actual,reference,divisor):
    rate,reference_rate=actual["rates"],reference["rates"]
    matrix,reference_matrix=actual["interaction"],reference["interaction"]
    flat=slopes(rate[:,:,0],divisor)
    reference_flat=slopes(reference_rate[:,:,0],divisor)
    probes=[]
    for k,name in enumerate(PROBE_NAMES):
        signal=slopes(rate[:,:,k],divisor)
        expected=slopes(reference_rate[:,:,k],divisor)
        delta=signal-expected
        probes.append(dict(probe=name,rate_relative_l2=float(relative(rate[:,:,k],reference_rate[:,:,k])),
            flux_relative_error=float(rate[:,:,k].sum()/reference_rate[:,:,k].sum()-1),
            slope_error_rms=float(np.sqrt(np.mean(delta**2))),slope_error_max=float(np.max(np.abs(delta))),
            slope_response_error_rms=float(np.sqrt(np.mean(((signal-flat)-(expected-reference_flat))**2))),
            slope_response_relative_l2=float(relative(signal-flat,expected-reference_flat)) if k>=2 else None))
    column_norm=np.linalg.norm(reference_matrix,axis=0)
    column_error=np.linalg.norm(matrix-reference_matrix,axis=0)
    relative_columns=np.divide(column_error,column_norm,out=np.full(column_error.shape,np.inf),where=column_norm!=0)
    singular=np.linalg.svd(matrix,compute_uv=False)
    expected_singular=np.linalg.svd(reference_matrix,compute_uv=False)
    result=dict(rate_max_relative_l2=max(p["rate_relative_l2"] for p in probes),
        rate_stack_relative_l2=float(relative(rate,reference_rate)),
        slope_response_max_relative_l2=max(p["slope_response_relative_l2"] for p in probes[2:]),
        slope_error_rms_max=max(p["slope_error_rms"] for p in probes),
        flat_slope_error_rms=probes[0]["slope_error_rms"],
        calibration_relative_l2=float(relative(matrix,reference_matrix)),
        calibration_max_column_relative_l2=float(relative_columns.max()),
        calibration_gain=float(np.sum(matrix*reference_matrix)/np.sum(reference_matrix**2)),
        calibration_spectrum_relative_l2=float(relative(singular,expected_singular)),
        calibration_rank_5percent=int(np.count_nonzero(singular>0.05*singular[0])),
        calibration_reference_rank_5percent=int(np.count_nonzero(expected_singular>0.05*expected_singular[0])),
        piston_invariance=float(relative(rate[:,:,1],rate[:,:,0])),
        photon_conservation=float(np.max(np.abs(rate.sum(axis=(0,1))/rate[:,:,0].sum()-1))),
        probes=probes,calibration_column_errors=relative_columns.tolist(),
        calibration_column_reference_norms=column_norm.tolist(),
        singular_values=singular.tolist())
    return result


def analyze(directory,output,extended=False):
    reference_record,reference=load_record(directory/"n1440-m256.toml")
    expected={(n,m) for n in (120,180,240,360,480,720) for m in (8,16,24,32,48,64)}
    expected.update(((1440,256),(960,256),(1440,128)))
    if extended:expected.update((n,m) for n in (120,240,480) for m in (96,128,256))
    available={tuple(map(int,p.stem[1:].split('-m'))) for p in directory.glob('n*-m*.toml')}
    assert available==expected, f"case coverage differs: missing={expected-available}, extra={available-expected}"
    records={}
    for path in sorted(directory.glob("n*-m*.toml")):
        record,arrays=load_record(path)
        assert record["full_calibration"] is True
        assert record["probes"]==list(PROBE_NAMES)
        assert record["support_pixels"]==716 and record["calibration_poke_m"]==2e-8
        assert arrays["rates"].shape==(64,64,13) and arrays["interaction"].shape==(1432,277)
        assert record["graph_sha256"]==reference_record["graph_sha256"]
        assert record["aos_revision"]==reference_record["aos_revision"]
        assert record["calibration_columns"]==list(range(1,278))
        assert record["divisor"]==reference_record["divisor"]
        assert record["harness_sha256"]==reference_record["harness_sha256"]
        result=compare(arrays,reference,reference_record["divisor"])
        result.update(resolution=record["resolution"],modulation_points=record["modulation_points"],fft_side=record["fft_side"])
        if "atmosphere_hil" in record:
            for endpoint in ("calibration_graph","calibration_hil","atmosphere_graph","atmosphere_hil"):
                stats=record[endpoint]
                assert stats["host_bytes"]==[0]*16
                assert np.median(stats["samples_ns"])==stats["p50_ns"]
                result[endpoint+"_us"]=stats["p50_ns"]/1000
        records[path.stem]=result
    output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(dict(reference="n1440-m256",numpy=np.__version__,python=platform.python_version(),cases=records),indent=2,allow_nan=False)+"\n")
    keys=["resolution","modulation_points","fft_side","rate_max_relative_l2","slope_response_max_relative_l2","calibration_relative_l2","calibration_max_column_relative_l2","calibration_gain","calibration_rank_5percent","atmosphere_graph_us","atmosphere_hil_us"]
    with (output/"summary.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=keys,extrasaction="ignore");writer.writeheader()
        for result in sorted(records.values(),key=lambda r:(r["resolution"],r["modulation_points"])):writer.writerow(result)
    return records



def analyze_spatial(directory,output,divisor):
    def load(path):
        record=tomllib.loads(path.read_text());r=record['rates']
        content=(directory/r['file']).read_bytes()
        assert hashlib.sha256(content).hexdigest()==r['sha256']
        values=np.frombuffer(gzip.decompress(content),dtype='<f4').reshape(r['shape'],order='F').astype(float)
        assert values.shape==(64,64,6) and np.isfinite(values).all()
        return record,values
    reference_record,reference=load(directory/'n1440-m256.toml')
    expected={(1440,256),(1920,256),(1440,512),(480,32),(180,64),(120,96),(240,96),(240,128),(480,128)}
    assert {tuple(map(int,p.stem[1:].split('-m'))) for p in directory.glob('*.toml')}==expected
    results={}
    reference_flat=slopes(reference[:,:,0],divisor)
    for path in sorted(directory.glob('*.toml')):
        record,rate=load(path)
        assert record['probes']==reference_record['probes']
        assert record['harness_sha256']==reference_record['harness_sha256']
        own_flat=slopes(rate[:,:,0],divisor)
        rows=[]
        for k,name in enumerate(record['probes']):
            signal=slopes(rate[:,:,k],divisor);expected_signal=slopes(reference[:,:,k],divisor)
            rows.append(dict(probe=name,rate_relative_l2=float(relative(rate[:,:,k],reference[:,:,k])),
                slope_error_rms=float(np.sqrt(np.mean((signal-expected_signal)**2))),
                reference_slope_response_rms=float(np.sqrt(np.mean((expected_signal-reference_flat)**2))),
                slope_response_error_rms=float(np.sqrt(np.mean(((signal-own_flat)-(expected_signal-reference_flat))**2))),
                slope_response_relative_l2=float(relative(signal-own_flat,expected_signal-reference_flat)) if k else None))
        results[path.stem]=rows
    output.mkdir(parents=True,exist_ok=True)
    (output/'spatial-frequency.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    return results

def plot_results(records,output):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"]="copper-convergence-20261021"
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    grids=(120,180,240,360,480,720)
    points=(8,16,24,32,48,64,96,128,256)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    metrics=(("rate_max_relative_l2","Photon-rate error"),
             ("slope_response_max_relative_l2","Slope-response error"),
             ("calibration_relative_l2","Calibration-matrix error"))
    for ax,(key,title) in zip(axes,metrics):
        values=np.full((len(grids),len(points)),np.nan)
        for i,n in enumerate(grids):
            for j,m in enumerate(points):
                record=records.get(f"n{n}-m{m}")
                if record:values[i,j]=100*record[key]
        im=ax.imshow(values,aspect="auto",origin="lower",norm=LogNorm(vmin=.01,vmax=400),cmap="viridis")
        ax.set(xticks=range(len(points)),xticklabels=points,yticks=range(len(grids)),yticklabels=grids,
               xlabel="Modulation points",ylabel="Pupil resolution",title=title)
        for i in range(len(grids)):
            for j in range(len(points)):
                if np.isfinite(values[i,j]):ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white" if values[i,j]<2 else "black")
    fig.colorbar(im,ax=axes,label="Relative error (%) against 1440² / 256")
    fig.suptitle("Copper convergence: fixed physical inputs, detector geometry and normalization")
    fig.savefig(output/"convergence.svg",metadata={"Date":None})
    fig.savefig(output/"convergence.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    candidates=[r for r in records.values() if "atmosphere_hil_us" in r]
    for ax,(key,title) in zip(axes,metrics):
        for n in grids:
            rows=[r for r in candidates if r["resolution"]==n]
            ax.scatter([r["atmosphere_hil_us"]/1000 for r in rows],[100*r[key] for r in rows],label=str(n),s=25)
        baseline=records["n480-m32"]
        ax.scatter([baseline["atmosphere_hil_us"]/1000],[100*baseline[key]],marker="*",s=160,color="red",label="Current preset")
        ax.set(xscale="log",yscale="log",xlabel="Captured noisy HIL cycle (ms)",ylabel="Error (%)",title=title)
        ax.grid(alpha=.2)
    axes[-1].legend(title="Pupil grid",fontsize=7)
    fig.savefig(output/"accuracy_latency.svg",metadata={"Date":None})
    plt.close(fig)
    (output/"plot-version.txt").write_text("matplotlib "+matplotlib.__version__+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--extended",action="store_true")
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--spatial",type=Path)
    args=parser.parse_args();records=analyze(args.directory,args.output,args.extended)
    if args.plot:plot_results(records,args.output)
    if args.spatial:
        reference=tomllib.loads((args.directory/"n1440-m256.toml").read_text())
        analyze_spatial(args.spatial,args.output,reference["divisor"])
