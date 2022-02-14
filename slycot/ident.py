#
#       ident.py
#
#       Copyright 2022 Ati Sharma <ati@agalmic.ltd>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License version 2 as
#       published by the Free Software Foundation.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

from . import _wrapper
from .exceptions import raise_if_slycot_error

import numpy as np


# subroutine ib01ad(meth,alg,jobd,batch,conct,ctrl,nobr,m,l,nsmp,u,ldu,y,ldy,n,r,ldr,sv,rcond,tol,iwork,dwork,ldwork,iwarn,info)
# subroutine ib01bd(meth,job,jobck,nobr,n,m,l,nsmpl,r,ldr,a,lda,c,ldc,b,ldb,d,ldd,q,ldq,ry,ldry,s,lds,k,ldk,tol,iwork,dwork,ldwork,bwork,iwarn,info)
# subroutine ib01cd(jobx0,comuse,job,n,m,l,nsmp,a,lda,b,ldb,c,ldc,d,ldd,u,ldu,y,ldy,x0,v,ldv,tol,iwork,dwork,ldwork,iwarn,info)


def ib01ad(nobr, u, y, r,
        rcond=0, tol=0,
        ctrl='N',
        meth='N',
        alg='F',
        batch='O',
        conct='N',
        jobd='N'):
    """ ?, ?, ? = ib01ad(?, ?, ?)

    To preprocess the input-output data for estimating the matrices
    of a linear time-invariant dynamical system and to find an
    estimate of the system order. The input-output data can,
    optionally, be processed sequentially.

    Parameters
    ----------

    meth :   str
             Specifies the subspace identification method to be used,
             as follows:
             = 'M':  MOESP  algorithm with past inputs and outputs;
             = 'N':  N4SID  algorithm.

    alg :    str
             Specifies the algorithm for computing the triangular
             factor R, as follows:
             = 'C': Cholesky algorithm applied to the correlation
                    matrix of the input-output data;
             = 'F': Fast QR algorithm;
             = 'Q': QR algorithm applied to the concatenated block
                    Hankel matrices.

    jobd :   str
             Specifies whether or not the matrices B and D should later
             be computed using the MOESP approach, as follows:
             = 'M': the matrices B and D should later be computed
                    using the MOESP approach;
             = 'N': the matrices B and D should not be computed using
                    the MOESP approach.
             This parameter is not relevant for METH = 'N'.

    batch :  str
             Specifies whether or not sequential data processing is to
             be used, and, for sequential processing, whether or not
             the current data block is the first block, an intermediate
             block, or the last block, as follows:
             = 'F':  the first block in sequential data processing;
             = 'I':  an intermediate block in sequential data
                     processing;
             = 'L':  the last block in sequential data processing;
             = 'O':  one block only (non-sequential data processing).
             NOTE that when  100  cycles of sequential data processing
                  are completed for  BATCH = 'I',  a warning is
                  issued, to prevent for an infinite loop.

    conct :  str
             Specifies whether or not the successive data blocks in
             sequential data processing belong to a single experiment,
             as follows:
             = 'C':  the current data block is a continuation of the
                     previous data block and/or it will be continued
                     by the next data block;
             = 'N':  there is no connection between the current data
                     block and the previous and/or the next ones.
             This parameter is not used if BATCH = 'O'.

    ctrl :  str
            Specifies whether or not the user's confirmation of the
            system order estimate is desired, as follows:
            = 'C':  user's confirmation;
            = 'N':  no confirmation.
            If  CTRL = 'C',  a reverse communication routine,  IB01OY,
            is indirectly called (by SLICOT Library routine IB01OD),
            and, after inspecting the singular values and system order
            estimate,  n,  the user may accept  n  or set a new value.
            IB01OY  is not called if CTRL = 'N'.

    NOBR    int
            The number of block rows,  s,  in the input and output
            block Hankel matrices to be processed.  NOBR > 0.
            (In the MOESP theory,  NOBR  should be larger than  n,
            the estimated dimension of state vector.)

    M       int
            The number of system inputs.  M >= 0.
            When M = 0, no system inputs are processed.

    L       int
            The number of system outputs.  L > 0.

    NSMP    int
            The number of rows of matrices  U  and  Y  (number of
            samples,  t). (When sequential data processing is used,
            NSMP  is the number of samples of the current data
            block.)
            NSMP >= 2*(M+L+1)*NOBR - 1,  for non-sequential
                                         processing;
            NSMP >= 2*NOBR,  for sequential processing.
            The total number of samples when calling the routine with
            BATCH = 'L'  should be at least  2*(M+L+1)*NOBR - 1.
            The  NSMP  argument may vary from a cycle to another in
            sequential data processing, but  NOBR, M,  and  L  should
            be kept constant. For efficiency, it is advisable to use
            NSMP  as large as possible.

    U       (LDU, M) array_like
            The leading NSMP-by-M part of this array must contain the
            t-by-m input-data sequence matrix  U,
            U = [u_1 u_2 ... u_m].  Column  j  of  U  contains the
            NSMP  values of the j-th input component for consecutive
            time increments.
            If M = 0, this array is not referenced.

    LDU     int
            The leading dimension of the array U.
            LDU >= NSMP, if M > 0;
            LDU >= 1,    if M = 0.

    Y       (LDY, L) array_like
            The leading NSMP-by-L part of this array must contain the
            t-by-l output-data sequence matrix  Y,
            Y = [y_1 y_2 ... y_l].  Column  j  of  Y  contains the
            NSMP  values of the j-th output component for consecutive
            time increments.

    LDY     int
            The leading dimension of the array Y.  LDY >= NSMP.

    R       array_like, optional
            ( LDR,2*(M+L)*NOBR )
            On exit, if ALG = 'C' and BATCH = 'F' or 'I', the leading
            2*(M+L)*NOBR-by-2*(M+L)*NOBR upper triangular part of this
            array contains the current upper triangular part of the
            correlation matrix in sequential data processing.
            If ALG = 'F' and BATCH = 'F' or 'I', the array R is not
            referenced.
            On exit, if INFO = 0, ALG = 'Q', and BATCH = 'F' or 'I',
            the leading 2*(M+L)*NOBR-by-2*(M+L)*NOBR upper triangular
            part of this array contains the current upper triangular
            factor R from the QR factorization of the concatenated
            block Hankel matrices. Denote  R_ij, i,j = 1:4,  the
            ij submatrix of  R,  partitioned by M*NOBR,  M*NOBR,
            L*NOBR,  and  L*NOBR  rows and columns.
            On exit, if INFO = 0 and BATCH = 'L' or 'O', the leading
            2*(M+L)*NOBR-by-2*(M+L)*NOBR upper triangular part of
            this array contains the matrix S, the processed upper
            triangular factor R from the QR factorization of the
            concatenated block Hankel matrices, as required by other
            subroutines. Specifically, let  S_ij, i,j = 1:4,  be the
            ij submatrix of  S,  partitioned by M*NOBR,  L*NOBR,
            M*NOBR,  and  L*NOBR  rows and columns. The submatrix
            S_22  contains the matrix of left singular vectors needed
            subsequently. Useful information is stored in  S_11  and
            in the block-column  S_14 : S_44.  For METH = 'M' and
            JOBD = 'M', the upper triangular part of  S_31  contains
            the upper triangular factor in the QR factorization of the
            matrix  R_1c = [ R_12'  R_22'  R_11' ]',  and  S_12
            contains the corresponding leading part of the transformed
            matrix  R_2c = [ R_13'  R_23'  R_14' ]'.  For  METH = 'N',
            the subarray  S_41 : S_43  contains the transpose of the
            matrix contained in  S_14 : S_34.
            The details of the contents of R need not be known if this
            routine is followed by SLICOT Library routine IB01BD.
            On entry, if ALG = 'C', or ALG = 'Q', and BATCH = 'I' or
            'L', the leading  2*(M+L)*NOBR-by-2*(M+L)*NOBR  upper
            triangular part of this array must contain the upper
            triangular matrix R computed at the previous call of this
            routine in sequential data processing. The array R need
            not be set on entry if ALG = 'F' or if BATCH = 'F' or 'O'.

    Returns
    -------
    N       int
            The estimated order of the system.
            If  CTRL = 'C',  the estimated order has been reset to a
            value specified by the user.

    R       (LDR, 2*(M+L)*NOBR) array_like
            See description of optional input parameter R.
            
    LDR     int
            The leading dimension of the array  R.
            LDR >= MAX( 2*(M+L)*NOBR, 3*M*NOBR ),
                                 for METH = 'M' and JOBD = 'M';
            LDR >= 2*(M+L)*NOBR, for METH = 'M' and JOBD = 'N' or
                                 for METH = 'N'.

    SV      (L*NOBR) array_like
            The singular values used to estimate the system order.

    Tolerances

    RCOND   DOUBLE PRECISION
            The tolerance to be used for estimating the rank of
            matrices. If the user sets  RCOND > 0,  the given value
            of  RCOND  is used as a lower bound for the reciprocal
            condition number;  an m-by-n matrix whose estimated
            condition number is less than  1/RCOND  is considered to
            be of full rank.  If the user sets  RCOND <= 0,  then an
            implicitly computed, default tolerance, defined by
            RCONDEF = m*n*EPS,  is used instead, where  EPS  is the
            relative machine precision (see LAPACK Library routine
            DLAMCH).
            This parameter is not used for  METH = 'M'.

    TOL     DOUBLE PRECISION
            Absolute tolerance used for determining an estimate of
            the system order. If  TOL >= 0,  the estimate is
            indicated by the index of the last singular value greater
            than or equal to  TOL.  (Singular values less than  TOL
            are considered as zero.) When  TOL = 0,  an internally
            computed default value,  TOL = NOBR*EPS*SV(1),  is used,
            where  SV(1)  is the maximal singular value, and  EPS  is
            the relative machine precision (see LAPACK Library routine
            DLAMCH). When  TOL < 0,  the estimate is indicated by the
            index of the singular value that has the largest
            logarithmic gap to its successor.

    Workspace

    IWORK   INTEGER array, dimension (LIWORK)
            LIWORK >= MAX(3,(M+L)*NOBR), if METH = 'N';
            LIWORK >= MAX(3,M+L), if METH = 'M' and ALG = 'F';
            LIWORK >= 3,   if METH = 'M' and ALG = 'C' or 'Q'.
            On entry with  BATCH = 'I'  or  BATCH = 'L',  IWORK(1:3)
            must contain the values of ICYCLE, MAXWRK, and NSMPSM
            set by the previous call of this routine.
            On exit with  BATCH = 'F'  or  BATCH = 'I',  IWORK(1:3)
            contains the values of ICYCLE, MAXWRK, and NSMPSM to be
            used by the next call of the routine.
            ICYCLE  counts the cycles for  BATCH = 'I'.
            MAXWRK  stores the current optimal workspace.
            NSMPSM  sums up the  NSMP  values for  BATCH <> 'O'.
            The first three elements of  IWORK  should be preserved
            during successive calls of the routine with  BATCH = 'F'
            or  BATCH = 'I',  till the final call with   BATCH = 'L'.

    DWORK   DOUBLE PRECISION array, dimension (LDWORK)
            On exit, if  INFO = 0,  DWORK(1) returns the optimal value
            of LDWORK,  and, for  METH = 'N',  and  BATCH = 'L'  or
            'O',  DWORK(2)  and  DWORK(3)  contain the reciprocal
            condition numbers of the triangular factors of the
            matrices  U_f  and  r_1  [6].
            On exit, if  INFO = -23,  DWORK(1)  returns the minimum
            value of LDWORK.
            Let
            k = 0,               if CONCT = 'N' and ALG = 'C' or 'Q';
            k = 2*NOBR-1,        if CONCT = 'C' and ALG = 'C' or 'Q';
            k = 2*NOBR*(M+L+1),  if CONCT = 'N' and ALG = 'F';
            k = 2*NOBR*(M+L+2),  if CONCT = 'C' and ALG = 'F'.
            The first (M+L)*k elements of  DWORK  should be preserved
            during successive calls of the routine with  BATCH = 'F'
            or  'I',  till the final call with  BATCH = 'L'.

    LDWORK  INTEGER
            The length of the array DWORK.
            LDWORK >= (4*NOBR-2)*(M+L), if ALG = 'C', BATCH = 'F' or
                            'I' and CONCT = 'C';
            LDWORK >= 1, if ALG = 'C', BATCH = 'F' or 'I' and
                            CONCT = 'N';
            LDWORK >= max((4*NOBR-2)*(M+L), 5*L*NOBR), if METH = 'M',
                            ALG = 'C', BATCH = 'L' and CONCT = 'C';
            LDWORK >= max((2*M-1)*NOBR, (M+L)*NOBR, 5*L*NOBR),
                            if METH = 'M', JOBD = 'M', ALG = 'C',
                             BATCH = 'O', or
                            (BATCH = 'L' and CONCT = 'N');
            LDWORK >= 5*L*NOBR, if METH = 'M', JOBD = 'N', ALG = 'C',
                             BATCH = 'O', or
                            (BATCH = 'L' and CONCT = 'N');
            LDWORK >= 5*(M+L)*NOBR+1, if METH = 'N', ALG = 'C', and
                            BATCH = 'L' or 'O';
            LDWORK >= (M+L)*2*NOBR*(M+L+3), if ALG = 'F',
                            BATCH <> 'O' and CONCT = 'C';
            LDWORK >= (M+L)*2*NOBR*(M+L+1), if ALG = 'F',
                            BATCH = 'F', 'I' and CONCT = 'N';
            LDWORK >= (M+L)*4*NOBR*(M+L+1)+(M+L)*2*NOBR, if ALG = 'F',
                            BATCH = 'L' and CONCT = 'N', or
                            BATCH = 'O';
            LDWORK >= 4*(M+L)*NOBR, if ALG = 'Q', BATCH = 'F', and
                            LDR >= NS = NSMP - 2*NOBR + 1;
            LDWORK >= max(4*(M+L)*NOBR, 5*L*NOBR), if METH = 'M',
                            ALG = 'Q', BATCH = 'O', and LDR >= NS;
            LDWORK >= 5*(M+L)*NOBR+1, if METH = 'N', ALG = 'Q',
                            BATCH = 'O', and LDR >= NS;
            LDWORK >= 6*(M+L)*NOBR, if ALG = 'Q', (BATCH = 'F' or 'O',
                            and LDR < NS), or (BATCH = 'I' or
                            'L' and CONCT = 'N');
            LDWORK >= 4*(NOBR+1)*(M+L)*NOBR, if ALG = 'Q', BATCH = 'I'
                            or 'L' and CONCT = 'C'.
            The workspace used for ALG = 'Q' is
                      LDRWRK*2*(M+L)*NOBR + 4*(M+L)*NOBR,
            where LDRWRK = LDWORK/(2*(M+L)*NOBR) - 2; recommended
            value LDRWRK = NS, assuming a large enough cache size.
            For good performance,  LDWORK  should be larger.

    Raises
    ------
    SlycotArithmeticError
        :info = 0:
                  successful exit;
        :info < 0:
                  if info = -i, the i-th argument had an illegal value;
        :info = 1:
                  a fast algorithm was requested (ALG = 'C', or 'F')
                  in sequential data processing, but it failed; the
                  routine can be repeatedly called again using the
                  standard QR algorithm;
        :info = 2:
                  the singular value decomposition (SVD) algorithm did
                  not converge.

    Warns
    -----
    SlycotResultWarning
    iwarn : int
        :iwarn = 0:
                  no warning;
        :iwarn = 1:
                  the number of 100 cycles in sequential data
                  processing has been exhausted without signaling
                  that the last block of data was get; the cycle
                  counter was reinitialized;
        :iwarn = 2:
                  a fast algorithm was requested (ALG = 'C' or 'F'),
                  but it failed, and the QR algorithm was then used
                  (non-sequential data processing);
        :iwarn = 3:
                  all singular values were exactly zero, hence  N = 0
                  (both input and output were identically zero);
        :iwarn = 4:
                  the least squares problems with coefficient matrix
                  U_f,  used for computing the weighted oblique
                  projection (for METH = 'N'), have a rank-deficient
                  coefficient matrix;
        :iwarn = 5:
                  the least squares problem with coefficient matrix
                  r_1  [6], used for computing the weighted oblique
                  projection (for METH = 'N'), has a rank-deficient
                  coefficient matrix.
            NOTE: the values 4 and 5 of IWARN have no significance
                  for the identification problem.

    Notes
    -----

    **Method**

    The procedure consists in three main steps, the first step being
    performed by one of the three algorithms included.

    1.a) For non-sequential data processing using QR algorithm, a
    t x 2(m+l)s  matrix H is constructed, where

         H = [ Uf'         Up'      Y'      ],  for METH = 'M',
                 s+1,2s,t    1,s,t   1,2s,t

         H = [ U'       Y'      ],              for METH = 'N',
                1,2s,t   1,2s,t

    and  Up     , Uf        , U      , and  Y        are block Hankel
           1,s,t    s+1,2s,t   1,2s,t        1,2s,t
    matrices defined in terms of the input and output data [3].
    A QR factorization is used to compress the data.
    The fast QR algorithm uses a QR factorization which exploits
    the block-Hankel structure. Actually, the Cholesky factor of H'*H
    is computed.

    1.b) For sequential data processing using QR algorithm, the QR
    decomposition is done sequentially, by updating the upper
    triangular factor  R.  This is also performed internally if the
    workspace is not large enough to accommodate an entire batch.

    1.c) For non-sequential or sequential data processing using
    Cholesky algorithm, the correlation matrix of input-output data is
    computed (sequentially, if requested), taking advantage of the
    block Hankel structure [7].  Then, the Cholesky factor of the
    correlation matrix is found, if possible.

    2) A singular value decomposition (SVD) of a certain matrix is
    then computed, which reveals the order  n  of the system as the
    number of "non-zero" singular values. For the MOESP approach, this
    matrix is  [ R_24'  R_34' ]' := R(ms+1:(2m+l)s,(2m+l)s+1:2(m+l)s),
    where  R  is the upper triangular factor  R  constructed by SLICOT
    Library routine  IB01MD.  For the N4SID approach, a weighted
    oblique projection is computed from the upper triangular factor  R
    and its SVD is then found.

    3) The singular values are compared to the given, or default TOL,
    and the estimated order  n  is returned, possibly after user's
    confirmation.

    **Numerical Aspects**

    The implemented method is numerically stable (when QR algorithm is
    used), reliable and efficient. The fast Cholesky or QR algorithms
    are more efficient, but the accuracy could diminish by forming the
    correlation matrix.
    The most time-consuming computational step is step 1:
                                       2
    The QR algorithm needs 0(t(2(m+l)s) ) floating point operations.
                                          2              3
    The Cholesky algorithm needs 0(2t(m+l) s)+0((2(m+l)s) ) floating
    point operations.
                                         2           3 2
    The fast QR algorithm needs 0(2t(m+l) s)+0(4(m+l) s ) floating
    point operations.
                                               3
    Step 2 of the algorithm requires 0(((m+l)s) ) floating point
    operations.

    **Further Comments**

    For ALG = 'Q', BATCH = 'O' and LDR < NS, or BATCH <> 'O', the
    calculations could be rather inefficient if only minimal workspace
    (see argument LDWORK) is provided. It is advisable to provide as
    much workspace as possible. Almost optimal efficiency can be
    obtained for  LDWORK = (NS+2)*(2*(M+L)*NOBR),  assuming that the
    cache size is large enough to accommodate R, U, Y, and DWORK.

    **Contributor**

    V. Sima, Katholieke Universiteit Leuven, Feb. 2000.

    **Revisions**

    August 2000, March 2005, May 2020.

    **Keywords**

    Cholesky decomposition, Hankel matrix, identification methods,
    multivariable systems, QR decomposition, singular value
    decomposition.

    References
    ----------

    [1] Verhaegen M., and Dewilde, P.
        Subspace Model Identification. Part 1: The output-error
        state-space model identification class of algorithms.
        Int. J. Control, 56, pp. 1187-1210, 1992.

    [2] Verhaegen M.
        Subspace Model Identification. Part 3: Analysis of the
        ordinary output-error state-space model identification
        algorithm.
        Int. J. Control, 58, pp. 555-586, 1993.

    [3] Verhaegen M.
        Identification of the deterministic part of MIMO state space
        models given in innovations form from input-output data.
        Automatica, Vol.30, No.1, pp.61-74, 1994.

    [4] Van Overschee, P., and De Moor, B.
        N4SID: Subspace Algorithms for the Identification of
        Combined Deterministic-Stochastic Systems.
        Automatica, Vol.30, No.1, pp. 75-93, 1994.

    [5] Peternell, K., Scherrer, W. and Deistler, M.
        Statistical Analysis of Novel Subspace Identification Methods.
        Signal Processing, 52, pp. 161-177, 1996.

    [6] Sima, V.
        Subspace-based Algorithms for Multivariable System
        Identification.
        Studies in Informatics and Control, 5, pp. 335-344, 1996.

    [7] Sima, V.
        Cholesky or QR Factorization for Data Compression in
        Subspace-based Identification ?
        Proceedings of the Second NICONET Workshop on ``Numerical
        Control Software: SLICOT, a Useful Tool in Industry'',
        December 3, 1999, INRIA Rocquencourt, France, pp. 75-80, 1999.
    """
    # input array size checks
    m = np.size(u, 1)
    l = np.size(y, 1)
    ldr = np.size(r, 0)
    if meth == 'M' and jobd == 'M':
        assert(ldr >= max(2*(m+l)*nobr, 3*m*nobr))
    elif (meth == 'M' and jobd == 'N') or meth == 'N':
        assert(ldr >= 2*(m+l)*nobr)
    hidden = ' (hidden by the wrapper)'
    arg_list = [
            'meth', 'alg', 'jobd', 'batch', 'conct', 'ctrl', 'nobr',
            'm'+hidden, 'l'+hidden, 'nsmp'+hidden, 'u', 'ldu'+hidden, 'y', 'ldy'+hidden,
            'n', 'r', 'ldr'+hidden,
            'sv',
            'rcond', 'tol',
            'iwork'+hidden, 'dwork'+hidden, 'ldwork'+hidden,
            'iwarn', 'info'
            ]
    n, r, sv, iwarn, info = _wrapper.ib01ad(
            meth, alg, jobd, batch, conct, ctrl, nobr,
            u, y, r,
            rcond, tol)
    raise_if_slycot_error([iwarn, info], arg_list, ib01ad.__doc__, locals())
    return n, r, sv, iwarn, info


def ib01bd():
    """
    """
    hidden = ' (hidden by the wrapper)'
    arg_list = []
    iwarn, info = _wrapper.ib01bd()
    raise_if_slycot_error(info, arg_list, ib01bd)
    return None


def ib01cd():
    """
    """
    hidden = ' (hidden by the wrapper)'
    arg_list = []
    out = _wrapper.ib01cd()
    raise_if_slycot_error(out, arg_list, ib01cd)
    return None


def ib03ad():
    """
    """
    hidden = ' (hidden by the wrapper)'
    arg_list = []
    out = _wrapper.ib03ad()
    raise_if_slycot_error(out, arg_list, ib03ad)
    return None


def ib03bd():
    """
    """
    hidden = ' (hidden by the wrapper)'
    arg_list = []
    out = _wrapper.ib03bd()
    raise_if_slycot_error(out, arg_list, ib03bd)
    return None
