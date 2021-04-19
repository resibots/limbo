import subprocess
import os
import re
import sys
from waflib.Configure import conf

# see : https://stackoverflow.com/questions/47878352/how-to-check-if-compiled-code-uses-sse-and-avx-instructions

avx_instructions = "(vmovapd|vmulpd|vaddpd|vsubpd|vfmadd213pd|vfmadd231pd|vfmadd132pd|vmulsd|vaddsd|vmosd|vsubsd|vbroadcastss|vbroadcastsd|vblendpd|vshufpd|vroundpd|vroundsd|vxorpd|vfnmadd231pd|vfnmadd213pd|vfnmadd132pd|vandpd|vmaxpd|vmovmskpd|vcmppd|vpaddd|vbroadcastf128|vinsertf128|vextractf128|vfmsub231pd|vfmsub132pd|vfmsub213pd|vmaskmovps|vmaskmovpd|vpermilps|vpermilpd|vperm2f128|vzeroall|vzeroupper|vpbroadcastb|vpbroadcastw|vpbroadcastd|vpbroadcastq|vbroadcasti128|vinserti128|vextracti128|vpminud|vpmuludq|vgatherdpd|vgatherqpd|vgatherdps|vgatherqps|vpgatherdd|vpgatherdq|vpgatherqd|vpgatherqq|vpmaskmovd|vpmaskmovq|vpermps|vpermd|vpermpd|vpermq|vperm2i128|vpblendd|vpsllvd|vpsllvq|vpsrlvd|vpsrlvq|vpsravd|vblendmpd|vblendmps|vpblendmd|vpblendmq|vpblendmb|vpblendmw|vpcmpd|vpcmpud|vpcmpq|vpcmpuq|vpcmpb|vpcmpub|vpcmpw|vpcmpuw|vptestmd|vptestmq|vptestnmd|vptestnmq|vptestmb|vptestmw|vptestnmb|vptestnmw|vcompresspd|vcompressps|vpcompressd|vpcompressq|vexpandpd|vexpandps|vpexpandd|vpexpandq|vpermb|vpermw|vpermt2b|vpermt2w|vpermi2pd|vpermi2ps|vpermi2d|vpermi2q|vpermi2b|vpermi2w|vpermt2ps|vpermt2pd|vpermt2d|vpermt2q|vshuff32x4|vshuff64x2|vshuffi32x4|vshuffi64x2|vpmultishiftqb|vpternlogd|vpternlogq|vpmovqd|vpmovsqd|vpmovusqd|vpmovqw|vpmovsqw|vpmovusqw|vpmovqb|vpmovsqb|vpmovusqb|vpmovdw|vpmovsdw|vpmovusdw|vpmovdb|vpmovsdb|vpmovusdb|vpmovwb|vpmovswb|vpmovuswb|vcvtps2udq|vcvtpd2udq|vcvttps2udq|vcvttpd2udq|vcvtss2usi|vcvtsd2usi|vcvttss2usi|vcvttsd2usi|vcvtps2qq|vcvtpd2qq|vcvtps2uqq|vcvtpd2uqq|vcvttps2qq|vcvttpd2qq|vcvttps2uqq|vcvttpd2uqq|vcvtudq2ps|vcvtudq2pd|vcvtusi2ps|vcvtusi2pd|vcvtusi2sd|vcvtusi2ss|vcvtuqq2ps|vcvtuqq2pd|vcvtqq2pd|vcvtqq2ps|vgetexppd|vgetexpps|vgetexpsd|vgetexpss|vgetmantpd|vgetmantps|vgetmantsd|vgetmantss|vfixupimmpd|vfixupimmps|vfixupimmsd|vfixupimmss|vrcp14pd|vrcp14ps|vrcp14sd|vrcp14ss|vrndscaleps|vrndscalepd|vrndscaless|vrndscalesd|vrsqrt14pd|vrsqrt14ps|vrsqrt14sd|vrsqrt14ss|vscalefps|vscalefpd|vscalefss|vscalefsd|valignd|valignq|vdbpsadbw|vpabsq|vpmaxsq|vpmaxuq|vpminsq|vpminuq|vprold|vprolvd|vprolq|vprolvq|vprord|vprorvd|vprorq|vprorvq|vpscatterdd|vpscatterdq|vpscatterqd|vpscatterqq|vscatterdps|vscatterdpd|vscatterqps|vscatterqpd|vpconflictd|vpconflictq|vplzcntd|vplzcntq|vpbroadcastmb2q|vpbroadcastmw2d|vexp2pd|vexp2ps|vrcp28pd|vrcp28ps|vrcp28sd|vrcp28ss|vrsqrt28pd|vrsqrt28ps|vrsqrt28sd|vrsqrt28ss|vgatherpf0dps|vgatherpf0qps|vgatherpf0dpd|vgatherpf0qpd|vgatherpf1dps|vgatherpf1qps|vgatherpf1dpd|vgatherpf1qpd|vscatterpf0dps|vscatterpf0qps|vscatterpf0dpd|vscatterpf0qpd|vscatterpf1dps|vscatterpf1qps|vscatterpf1dpd|vscatterpf1qpd|vfpclassps|vfpclasspd|vfpclassss|vfpclasssd|vrangeps|vrangepd|vrangess|vrangesd|vreduceps|vreducepd|vreducess|vreducesd|vpmovm2d|vpmovm2q|vpmovm2b|vpmovm2w|vpmovd2m|vpmovq2m|vpmovb2m|vpmovw2m|vpmullq|vpmadd52luq|vpmadd52huq|v4fmaddps|v4fmaddss|v4fnmaddps|v4fnmaddss|vp4dpwssd|vp4dpwssds|vpdpbusd|vpdpbusds|vpdpwssd|vpdpwssds|vpcompressb|vpcompressw|vpexpandb|vpexpandw|vpshld|vpshldv|vpshrd|vpshrdv|vpopcntd|vpopcntq|vpopcntb|vpopcntw|vpshufbitqmb|gf2p8affineinvqb|gf2p8affineqb|gf2p8mulb|vpclmulqdq|vaesdec|vaesdeclast|vaesenc|vaesenclast)"

def _obj_path(obj_name, path_list):
    for p in path_list:
        name = os.path.join(p, obj_name)
        if os.path.isfile(name):
            return name
    return None

def test_avx(obj_name, path_list):
    p = _obj_path(obj_name, path_list)
    if p == None:
        print("WARNING:", obj_name, " not found")
        return False
    asm = subprocess.check_output(["objdump", "-d", p], shell=False)
    regex = re.compile(avx_instructions)
    m = regex.search(asm.decode())
    if m:
        return True
    else:
        return False

@conf
def check_avx(conf, lib, required = [], lib_type = 'shared'):
    paths = conf.env['LIBPATH_' + lib.upper()]
    libs = conf.env['LIB_' + lib.upper()]
    if sys.platform == 'darwin':
        ext = '.dylib'
    else:
        ext = '.so'
    if lib_type == 'static':
        libs = conf.env['STLIB_' + lib.upper()]
        ext = '.a'
    if not isinstance(libs, list):
        libs = [libs]
    if not isinstance(paths, list):
        paths = [paths]
    if len(libs) == 0:
        return False
    failed = False
    for l in libs:
        if l in required or len(required) == 0:
            res = test_avx('lib' + l + ext, paths)
            conf.start_msg('AVX compilation of ' + l)
            if not res:
                conf.end_msg('no', 'YELLOW')
                failed = True
            else:
                conf.end_msg('yes', 'GREEN')
    return not failed
