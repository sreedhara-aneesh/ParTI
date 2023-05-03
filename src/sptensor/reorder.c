#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "ParTI.h"
#include "sptensor.h"

/*Interface to everything in this file is orderit(.., ..)*/

/*function declarations*/
void sptLexiOrderPerMode(sptSparseTensor * tsr, sptIndex mode, sptIndex ** orgIds, int tk);
void sptBFSLike(sptSparseTensor * tsr, sptIndex ** newIndices);

// My own declarations
void spt_lexi_order(sptSparseTensor* tsr, sptIndex** newIndices, sptIndex iterations);
void spt_bfs_mcs_order(sptSparseTensor * tensor, sptIndex ** new_indices);
void spt_lexi_order_for_mode(sptSparseTensor* tensor, sptIndex mode, sptIndex* og_indices);

static double u_seconds(void)
{
    struct timeval tp;

    gettimeofday(&tp, NULL);

    return (double) tp.tv_sec + (double) tp.tv_usec / 1000000.0;

};

void sptIndexRenumber(sptSparseTensor * tsr, sptIndex ** newIndices, int renumber, sptIndex iterations, int tk)
{
    /*
     newIndices is of size [nmodes][ndims[modes]] and assumed to be allocted.
     It will be overwritten. No need to initialize.

     We will need to reshuffle nonzeros. In order to not to touch tsr, we copy the indices of nonzeros
     to a local variable coords. This is sort of transposed wrt tsr: its size is nnz * n, instead of n * nnz used in tsr.
     */
    sptIndex const nmodes = tsr->nmodes;
    sptNnzIndex const nnz = tsr->nnz;

    sptIndex i, m;
    sptNnzIndex z;
    sptIndex its;

    // TODO: user is currently unable to select renumber choices past 1, 2

    if (renumber == 1) {    /* Lexi-order renumbering */
        /* copy the indices */
        sptSparseTensor tsr_temp;
        sptCopySparseTensor(&tsr_temp, tsr, tk);

        sptIndex ** orgIds = (sptIndex **) malloc(sizeof(sptIndex*) * nmodes);

        for (m = 0; m < nmodes; m++)
        {
            orgIds[m] = (sptIndex *) malloc(sizeof(sptIndex) * tsr->ndims[m]);
            // #pragma omp parallel for num_threads(tk) private(i)
            for (i = 0; i < tsr->ndims[m]; i++)
                orgIds[m][i] = i;
        }

        for (its = 0; its < iterations; its++)
        {
            printf("[Lexi-order] Optimizing the numbering for its %u\n", its+1);
            for (m = 0; m < nmodes; m++)
                sptLexiOrderPerMode(&tsr_temp, m, orgIds, tk);
        }

        /* compute newIndices from orgIds. Reverse perm */
        for (m = 0; m < nmodes; m++)
            for (i = 0; i < tsr->ndims[m]; i++)
                newIndices[m][orgIds[m][i]] = i;

        sptFreeSparseTensor(&tsr_temp);
        for (m = 0; m < nmodes; m++)
            free(orgIds[m]);
        free(orgIds);

    } else if (renumber == 2) {    /* BFS-like renumbering */
        /*
         REMARK (10 May 2018): this is the old bfs-like kind of thing. I hoped it would reduce the number of iterations,
         but on a few cases it did not help much. Just leaving it in case we want to use it.
         */
        printf("[BFS-like]\n");
        sptBFSLike(tsr, newIndices);
    } else if (renumber == 4) {

        printf("MY IMPLEMENTATION: LEXI_ORDER\n");

        // Copy indices
        sptSparseTensor tsr_temp;
        sptCopySparseTensor(&tsr_temp, tsr, tk);

        // Run lexi_order
        spt_lexi_order(&tsr_temp, newIndices, iterations);

        // Free stuff
        sptFreeSparseTensor(&tsr_temp);
    } else if (renumber == 5) {
        printf("MY IMPLEMENTATION: BFS_MCS_ORDER\n");
        spt_bfs_mcs_order(tsr, newIndices);
    }

}


static void lexOrderThem( sptNnzIndex m, sptIndex n, sptNnzIndex *ia, sptIndex *cols, sptIndex *cprm)
{

//    TODO: get rid of this debug info
//    printf("\nm: %d\n", (int) m);
//    printf("n: %d\n", (int) n);
//
//    printf("ROWS:\n");
//    for (int i = 0; i < (int) m + 1; i++) {
//        printf("\t%f\n", (float) ia[i + 1]);
//    }
//
//    printf("COLS:\n");
//    for (int i = 0; i < (int) n + 1; i++) {
//        printf("\t%d\n", (int) cols[i + 1]);
//    }

    /*m, n are the num of rows and cols, respectively. We lex order cols,
     given rows.

     BU notes as of 4 May 2018: I am hoping that I will not be asked the details of this function, and its memory use;) A quick and dirty update from something else I had since some time. I did not think through if the arrays could be reduced. Right now we have 10 arrays of size n each (where n is the length of a single dimension of the tensor.
     */

    sptNnzIndex *flag, j, jcol, jend;
    sptIndex *svar,  *var, numBlocks;
    sptIndex *prev, *next, *sz, *setnext, *setprev, *tailset;

    sptIndex *freeIdList, freeIdTop;

    sptIndex k, s, acol;

    sptIndex firstset, set, pos;

    svar = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    flag = (sptNnzIndex*) calloc(sizeof(sptNnzIndex),(n+2));
    var  = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    prev = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    next = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    sz   = (sptIndex*) calloc(sizeof(sptIndex),(n+2));
    setprev = (sptIndex*)calloc(sizeof(sptIndex),(n+2));
    setnext = (sptIndex*)calloc(sizeof(sptIndex),(n+2));
    tailset = (sptIndex*)calloc(sizeof(sptIndex),(n+2));
    freeIdList = (sptIndex*)calloc(sizeof(sptIndex),(n+2));

    next[1] = 2;
    prev[0] =  prev[1] = 0;
    next[n] = 0;
    prev[n] = n-1;
    svar[1] = svar[n] = 1;
    flag[1] = flag[n] = flag[n+1] = 0;
    cprm[1] = cprm[n] = 2 * n ;
    setprev[1] = setnext[1] = 0;

    for(sptIndex jj = 2; jj<=n-1; jj++)/*init all in a single svar*/
    {
        svar[jj] = 1;
        next[jj] = jj+1;
        prev[jj] = jj-1;
        flag[jj] = 0;
        sz[jj] = 0;
        setprev[jj] = setnext[jj] = 0;
        cprm[jj] = 2 * n;
    }
    var[1] = 1;
    sz[1] = n;
    sz[n] = sz[n+1] =  0;

    setprev[n] = setnext[n] = 0;
    setprev[n+1] = setnext[n+1] = 0;

    tailset[1] = n;

    firstset = 1;
    freeIdList[0] = 0;

    for(sptIndex jj= 1; jj<=n; jj++)
        freeIdList[jj] = jj+1;/*1 is used as a set id*/

    freeIdTop = 1;
    for(j=1; j<=m; j++)
    {
        jend = ia[j+1]-1;
        for(jcol = ia[j]; jcol <= jend ; jcol++)
        {
            acol= cols[jcol];
            s = svar[acol];
            if( flag[s] < j)/*first occurence of supervar s in j*/
            {
                flag[s] = j;
                if(sz[s] == 1 && tailset[s] != acol)
                {
                    printf("this should not happen (sz 1 but tailset not ok)\n");
                    exit(12);
                }
                if(sz[s] > 1)
                {
                    sptIndex newId;
                    /*remove acol from s*/
                    if(tailset[s] == acol) tailset[s] = prev[acol];

                    next[prev[acol]] = next[acol];
                    prev[next[acol]] = prev[acol];

                    sz[s] = sz[s] - 1;
                    /*create a new supervar ns=newId
                     and make i=acol its only var*/
                    if(freeIdTop == n+1) {
                        printf("this should not happen (no index)\n");
                        exit(12);
                    }
                    newId = freeIdList[freeIdTop++];
                    svar[acol] = newId;
                    var[newId] = acol;
                    flag[newId] = j;
                    sz[newId ] = 1;
                    next[acol] = 0;
                    prev[acol] = 0;
                    var[s] = acol;
                    tailset[newId] = acol;

                    setnext[newId] = s;
                    setprev[newId] = setprev[s];
                    if(setprev[s])
                        setnext[setprev[s]] = newId;
                    setprev[s] = newId;

                    if(firstset == s)
                        firstset = newId;

                }
            }
            else/*second or later occurence of s for row j*/
            {
                k = var[s];
                svar[acol] = svar[k];

                /*remove acol from its current chain*/
                if (tailset[s] == acol) tailset[s] = prev[acol];

                next[prev[acol]] = next[acol];
                prev[next[acol]] = prev[acol];

                sz[s] = sz[s] - 1;
                if (sz[s] == 0)/*s is a free id now..*/
                {

                    freeIdList[--freeIdTop] = s; /*add s to the free id list*/

                    if (setnext[s])
                        setprev[setnext[s]] = setprev[s];
                    if (setprev[s])
                        setnext[setprev[s]] = setnext[s];

                    setprev[s] = setnext[s] = 0;
                    tailset[s] = 0;
                    var[s] = 0;
                    flag[s] = 0;
                }
                /*add to chain containing k (as the last element)*/
                prev[acol] = tailset[svar[k]];
                next[acol] = 0;/*BU next[tailset[svar[k]]];*/
                next[tailset[svar[k]]] = acol;
                tailset[svar[k]] = acol;
                sz[svar[k]] = sz[svar[k]] + 1;
            }
        }
    }

    pos = 1;
    numBlocks = 0;
    for(set = firstset; set != 0; set = setnext[set])
    {
        sptIndex item = tailset[set];
        sptIndex headset = 0;
        numBlocks ++;

        while(item != 0 )
        {
            headset = item;
            item = prev[item];
        }
        /*located the head of the set. output them (this is for keeping the initial order*/
        while(headset)
        {
            cprm[pos++] = headset;
            headset = next[headset];
        }
    }

    free(tailset);
    free(sz);
    free(next);
    free(prev);
    free(var);
    free(flag);
    free(svar);
    free(setnext);
    free(setprev);
    if(pos-1 != n){
        printf("**************** Error ***********\n");
        printf("something went wrong and we could not order everyone\n");
        exit(12);
    }

    return ;
}


/**************************************************************/
#define myAbs(x) (((x) < 0) ? -(x) : (x))

void sptLexiOrderPerMode(sptSparseTensor * tsr, sptIndex mode, sptIndex ** orgIds, int tk)
{
    sptIndexVector * inds = tsr->inds;
    sptNnzIndex const nnz = tsr->nnz;
    sptIndex const nmodes = tsr->nmodes;
    sptIndex * ndims = tsr->ndims;
    sptIndex const mode_dim = ndims[mode];
    sptNnzIndex * rowPtrs = NULL;
    sptIndex * colIds = NULL;
    sptIndex * cprm = NULL, * invcprm = NULL, * saveOrgIds = NULL;
    sptNnzIndex atRowPlus1, mtxNrows, mtrxNnz;
    sptIndex * mode_order = (sptIndex *) malloc (sizeof(sptIndex) * (nmodes - 1));

    sptIndex c;
    sptNnzIndex z;
    double t1, t0;

    t0 = u_seconds();
    sptIndex i = 0;
    for(sptIndex m = 0; m < nmodes; ++m) {
        if (m != mode) {
            mode_order[i] = m;
            ++ i;
        }
    }
    sptSparseTensorSortIndexExceptSingleMode(tsr, 1, mode_order, tk);
    t1 = u_seconds()-t0;
    printf("mode %u, sort time %.2f\n", mode, t1);

    /* we matricize this (others x thisDim), whose columns will be renumbered */
    /* on the matrix all arrays are from 1, and all indices are from 1. */

    rowPtrs = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * (nnz + 2)); /*large space*/
    colIds = (sptIndex *) malloc(sizeof(sptIndex) * (nnz + 2)); /*large space*/

    if(rowPtrs == NULL || colIds == NULL)
    {
        printf("could not allocate.exiting \n");
        exit(12);
    }

    rowPtrs[0] = 0; /* we should not access this, that is why. */
    rowPtrs [1] = 1;
    colIds[1] = inds[mode].data[0] + 1;
    atRowPlus1 = 2;
    mtrxNnz = 2;/* start filling from the second element */

    t0 = u_seconds();
    for (z = 1; z < nnz; z++)
    {
        int cmp_res;
        cmp_res = spt_SparseTensorCompareIndicesExceptSingleMode(tsr, z, tsr, z-1, mode_order);
        if(cmp_res != 0)
            rowPtrs[atRowPlus1++] = mtrxNnz; /* close the previous row and start a new one. */

        colIds[mtrxNnz ++] = inds[mode].data[z] + 1;
    }
    rowPtrs[atRowPlus1] = mtrxNnz;
    mtxNrows = atRowPlus1-1;
    t1 =u_seconds()-t0;
    printf("mode %u, create time %.2f\n", mode, t1);

    rowPtrs = realloc(rowPtrs, (sizeof(sptNnzIndex) * (mtxNrows + 2)));
    cprm = (sptIndex *) malloc(sizeof(sptIndex) * (mode_dim + 1));
    invcprm = (sptIndex *) malloc(sizeof(sptIndex) * (mode_dim + 1));
    saveOrgIds = (sptIndex *) malloc(sizeof(sptIndex) * (mode_dim + 1));

    t0 = u_seconds();
    lexOrderThem(mtxNrows, mode_dim, rowPtrs, colIds, cprm);
    t1 =u_seconds()-t0;
    printf("mode %u, lexorder time %.2f\n", mode, t1);

    /* update orgIds and modify coords */
    for (c=0; c < mode_dim; c++)
    {
        invcprm[cprm[c+1]-1] = c;
        saveOrgIds[c] = orgIds[mode][c];
    }
    for (c=0; c < mode_dim; c++)
        orgIds[mode][c] = saveOrgIds[cprm[c+1]-1];

    /* rename the dim component of nonzeros */
    for (z = 0; z < nnz; z++)
        inds[mode].data[z] = invcprm[inds[mode].data[z]];

    free(saveOrgIds);
    free(invcprm);
    free(cprm);
    free(colIds);
    free(rowPtrs);
    free(mode_order);
}

/**************************************************************/

typedef struct{
    sptIndex nvrt; /* number of vertices. This nvrt = n_0 + n_1 + ... + n_{d-1} for a d-dimensional tensor
                   where the ith dimension is of size n_i.*/
    sptNnzIndex *vptrs, *vHids; /*starts of hedges containing vertices, and the ids of the hedges*/

    sptNnzIndex nhdg; /*this will be equal to the number of nonzeros in the tensor*/
    sptNnzIndex *hptrs, *hVids; /*starts of vertices in the hedges, and the ids of the vertices*/
} basicHypergraph;

static void allocateHypergraphData(basicHypergraph *hg, sptIndex nvrt, sptNnzIndex nhdg, sptNnzIndex npins)
{
    hg->nvrt = nvrt;
    hg->vptrs = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * (nvrt+1));
    hg->vHids = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * npins);

    hg->nhdg = nhdg;
    hg->hptrs = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * (nhdg+1));
    hg->hVids = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * npins);
}


static void freeHypergraphData(basicHypergraph *hg)
{
    hg->nvrt = 0;
    if (hg->vptrs) free(hg->vptrs);
    if (hg->vHids) free(hg->vHids);

    hg->nhdg = 0;
    if (hg->hptrs) free(hg->hptrs);
    if (hg->hVids) free(hg->hVids);
}


static void setVList(basicHypergraph *hg)
{
    /*PRE: We assume hg->hptrs and hg->hVids are set; hg->nvrts is set, and
     hg->vptrs and hg->vHids are allocated appropriately.
     */

    sptNnzIndex j, h, v, nhdg = hg->nhdg;

    sptIndex nvrt = hg->nvrt;

    /*vertices */
    sptNnzIndex *vptrs = hg->vptrs, *vHids = hg->vHids;
    /*hyperedges*/
    sptNnzIndex *hptrs = hg->hptrs, *hVids = hg->hVids;

    for (v = 0; v <= nvrt; v++)
        vptrs[v] = 0;

    for (h = 0; h < nhdg; h++)
    {
        for (j = hptrs[h]; j < hptrs[h+1]; j++)
        {
            v = hVids[j];
            vptrs[v] ++;
        }
    }
    for (v=1; v <= nvrt; v++)
        vptrs[v] += vptrs[v-1];

    for (h = nhdg; h >= 1; h--)
    {
        for (j = hptrs[h-1]; j < hptrs[h]; j++)
        {
            v = hVids[j];
            vHids[--(vptrs[v])] = h-1;
        }
    }
}

static void fillHypergraphFromCoo(basicHypergraph *hg, sptIndex nm, sptNnzIndex nnz, sptIndex *ndims, sptIndexVector * inds)
{

    sptIndex  totalSizes;
    sptNnzIndex h, toAddress;
    sptIndex *dimSizesPrefixSum;

    sptIndex i;

    dimSizesPrefixSum = (sptIndex *) malloc(sizeof(sptIndex) * (nm+1));
    totalSizes = 0;
    for (i=0; i < nm; i++)
    {
        dimSizesPrefixSum[i] = totalSizes;
        totalSizes += ndims[i];
    }
    printf("allocating hyp %u %lu\n", nm, nnz);

    allocateHypergraphData(hg, totalSizes, nnz, nnz * nm);

    toAddress = 0;
    for (h = 0; h < nnz; h++)
    {
        hg->hptrs[h] = toAddress;
        for (i = 0;  i < nm; i++)
            hg->hVids[toAddress + i] = dimSizesPrefixSum[i] + inds[i].data[h];
        toAddress += nm;
    }
    hg->hptrs[hg->nhdg] = toAddress;

    setVList(hg);
    free(dimSizesPrefixSum);
}

static inline sptIndex locateVertex(sptNnzIndex indStart, sptNnzIndex indEnd, sptNnzIndex *lst, sptNnzIndex sz)
{
    sptNnzIndex i;
    for (i = 0; i < sz; i++)
        if(lst[i] >= indStart && lst[i] <= indEnd)
            return lst[i];

    printf("could not locate in a hyperedge !!!\n");
    exit(1);
    return sz+1;
}

#define SIZEV( vid ) vptrs[(vid)+1]-vptrs[(vid)]
static void heapIncreaseKey(sptIndex *heapIds, sptNnzIndex *key, sptNnzIndex *vptrs, sptIndex sz, sptIndex id, sptIndex *inheap, sptNnzIndex newKey)
{

    sptIndex i = inheap[id]; /*location in heap*/
    if( i > 0 && i <=sz )
    {
        key[id] = newKey;

        while ((i>>1)>0 && ( (key[id] > key[heapIds[i>>1]]) ||
                             (key[id] == key[heapIds[i>>1]] && SIZEV(id) > SIZEV(heapIds[i>>1])))
                )
        {
            heapIds[i] = heapIds[i>>1];
            inheap[heapIds[i]] = i;
            i = i>>1;
        }
        heapIds[i] = id;
        inheap[id] = i;
    }
}


static void heapify(sptIndex *heapIds, sptNnzIndex *key, sptNnzIndex *vptrs, sptIndex sz, sptIndex i,  sptIndex *inheap)
{
    sptIndex largest, j, l,r, tmp;

    largest = j = i;
    while(j<=sz/2)
    {
        l = 2*j;
        r = 2*j + 1;

        if ( (key[heapIds[l]] > key[heapIds[j]] ) ||
             (key[heapIds[l]] == key[heapIds[j]]  && SIZEV(heapIds[l]) < SIZEV(heapIds[j]) )
                )
            largest = l;
        else
            largest = j;

        if (r<=sz && (key[heapIds[r]]>key[heapIds[largest]] ||
                      (key[heapIds[r]]==key[heapIds[largest]] && SIZEV(heapIds[r]) < SIZEV(heapIds[largest])))
                )
            largest = r;

        if (largest != j)
        {
            tmp = heapIds[largest];
            heapIds[largest] = heapIds[j];
            inheap[heapIds[j]] = largest;

            heapIds[j] = tmp;
            inheap[heapIds[j]] = j;
            j = largest;
        }
        else
            break;
    }
}

static sptIndex heapExtractMax(sptIndex *heapIds, sptNnzIndex *key, sptNnzIndex *vptrs, sptIndex *sz, sptIndex *inheap)
{
    sptIndex maxind ;
    if (*sz < 1){
        printf("Error: heap underflow\n"); exit(12);
    }
    maxind = heapIds[1];
    heapIds[1] = heapIds[*sz];
    inheap[heapIds[1]] = 1;

    *sz = *sz - 1;
    inheap[maxind] = 0;

    heapify(heapIds, key, vptrs, *sz, 1, inheap);
    return maxind;

}

static void heapBuild(sptIndex *heapIds, sptNnzIndex *key, sptNnzIndex *vptrs, sptIndex sz, sptIndex *inheap)
{
    sptIndex i;
    for (i=sz/2; i>=1; i--)
        heapify(heapIds, key, vptrs, sz, i, inheap);
}

static void orderforHiCOOaDim(basicHypergraph *hg, sptIndex *newIndicesHg, sptIndex indStart, sptIndex indEnd)
{
    /* we re-order the vertices of the hypergraph with ids in the range [indStart, indEnd]*/

    sptIndex i, v, heapSz, *inHeap, *heapIds;
    sptNnzIndex j, jj, hedge, hedge2, k, w, ww;
    sptNnzIndex *vptrs = hg->vptrs, *vHids = hg->vHids, *hptrs = hg->hptrs, *hVids = hg->hVids;

    sptNnzIndex *keyvals, newKeyval;
    int *markers, mark;

    mark = 0;

    heapIds = (sptIndex*) malloc(sizeof(sptIndex) * (indEnd-indStart + 2));
    inHeap = (sptIndex*) malloc(sizeof(sptIndex) * hg->nvrt);/*this is large*/
    keyvals = (sptNnzIndex *) malloc(sizeof(sptNnzIndex) * hg->nvrt);
    markers = (int*) malloc(sizeof(int)* hg->nvrt);

    heapSz = 0;

    for (i = indStart; i<=indEnd; i++)
    {
        keyvals[i] = 0;
        heapIds[++heapSz] = i;
        inHeap[i] = heapSz;
        markers[i] = -1;
    }
    heapBuild(heapIds, keyvals, vptrs, heapSz, inHeap);

    for (i = indStart; i <= indEnd; i++)
    {
        v = heapExtractMax(heapIds, keyvals, vptrs, &heapSz, inHeap);
        newIndicesHg[v] = i;
        markers[v] = mark;
        for (j = vptrs[v]; j < vptrs[v+1]; j++)
        {
            hedge = vHids[j];
            for (k = hptrs[hedge]; k < hptrs[hedge+1]; k++)
            {
                w = hVids[k];
                if (markers[w] != mark)
                {
                    markers[w] = mark;
                    for(jj = vptrs[w]; jj < vptrs[w+1]; jj++)
                    {
                        hedge2 = vHids[jj];
                        ww = locateVertex(indStart, indEnd, hVids + hptrs[hedge2], hptrs[hedge2+1]-hptrs[hedge2]);
                        if( inHeap[ww] )
                        {
                            newKeyval = keyvals[ww] + 1;
                            heapIncreaseKey(heapIds, keyvals, vptrs, heapSz, ww, inHeap, newKeyval);
                        }
                    }
                }
            }
        }
    }

    free(markers);
    free(keyvals);
    free(inHeap);
    free(heapIds);
}

/**************************************************************/
void sptBFSLike(sptSparseTensor * tsr, sptIndex ** newIndices)
{
    /*PRE: newIndices is allocated

     POST:
     newIndices[0][0...n_0-1] gives the new ids for dim 0
     newIndices[1][0...n_1-1] gives the new ids for dim 1
     ...
     newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1

     This implements a simple idea close to BFS/Cuthill-McKee/Maximum cardinality search.
     */

    sptIndex const nmodes = tsr->nmodes;
    sptNnzIndex const nnz = tsr->nnz;
    sptIndex * ndims = tsr->ndims;
    sptIndexVector * inds = tsr->inds;

    sptIndex *dimsPrefixSum;
    basicHypergraph hg;
    sptIndex *newIndicesHg;
    sptIndex d, i;

    dimsPrefixSum = (sptIndex*) calloc(nmodes, sizeof(sptIndex));
    for (d = 1; d < nmodes; d++)
        dimsPrefixSum[d] = ndims[d-1] + dimsPrefixSum[d-1];

    fillHypergraphFromCoo(&hg, nmodes,  nnz, ndims, inds);

    newIndicesHg = (sptIndex*) malloc(sizeof(sptIndex) * hg.nvrt);
    for (i = 0; i < hg.nvrt; i++)
        newIndicesHg[i] = i;

    for (d = 0; d < nmodes; d++) /*order d*/
        orderforHiCOOaDim(&hg, newIndicesHg, dimsPrefixSum[d], dimsPrefixSum[d] + ndims[d]-1);

    /*copy from newIndices to newIndicesOut*/
    for (d = 0; d < nmodes; d++)
        for (i = 0; i < ndims[d]; i++)
            newIndices[d][i] = newIndicesHg[dimsPrefixSum[d] + i] - dimsPrefixSum[d];

    free(newIndicesHg);
    freeHypergraphData(&hg);
    free(dimsPrefixSum);

}
/********************** Internals end *************************/

/********************** MY BFS_MCS_ORDER IMPLEMENTATION START *************************/

/**
 * Hyperedge used for Hypergraph for bfs_mcs_order
 */
typedef struct {
    /** Number of vertices connected by edge */
    sptNnzIndex num_vertices;
    /** Array of vertices connected by edge */
    sptNnzIndex* vertices;
} Hyperedge;

/**
 * Hypergraph used for bfs_mcs_order
 */
typedef struct {
    /** Number of hyperedges */
    sptNnzIndex num_edges;
    /** Array of hyperedges */
    Hyperedge* edges;
} Hypergraph;

/**
 * Node used for MaxHeap for bfs_mcs_order
 */
typedef struct {
    /**
     * Index number that this node correlates to.
     * NOT the index in the array of nodes in MaxHeap.
     */
    sptNnzIndex index;
    /** The primary key/weight. */
    sptNnzIndex prim_key;
    /**
     * The secondary key/weight.
     * Due to specific implementation reasons, this is inverted.
     * So, a sec_key of 1 would be greater than 2.
     */
    sptNnzIndex sec_key;
} MaxHeapNode;

/**
 * MaxHeap used for bfs_mcs_order
 */
typedef struct {
    /** Capacity of the heap. */
    sptNnzIndex capacity;
    /** Size of the heap. */
    sptNnzIndex size;
    /** Array of nodes in the heap. */
    MaxHeapNode* nodes;
} MaxHeap;

/**
 * Compare 2 MaxHeapNodes
 * @param a node a
 * @param b node b
 * @return 1 if a should be above b
 * @return -1 if b should be above a
 */
int compare_max_heap_nodes(MaxHeapNode* a, MaxHeapNode* b) {
    if (a->prim_key > b->prim_key) return 1;
    if (a->prim_key == b->prim_key && a->sec_key < b->sec_key) return 1;
    return -1;
}

/**
 * Create a Hypergraph from a given tensor.
 * @param tensor sparse tensor
 * @return hypergraph
 */
Hypergraph* create_hypergraph_from_spt_sparse_tensor(sptSparseTensor* tensor) {
    Hypergraph* hypergraph = malloc(sizeof(Hypergraph));
    hypergraph->num_edges = tensor->nnz;
    hypergraph->edges = (Hyperedge*) malloc(hypergraph->num_edges * sizeof(Hyperedge));
    for (sptNnzIndex i = 0; i < tensor->nnz; i++) {
        Hyperedge edge;
        edge.num_vertices = tensor->nmodes;
        edge.vertices = (sptNnzIndex*) malloc(edge.num_vertices * sizeof(sptNnzIndex));
        for (sptNnzIndex j = 0; j < tensor->nmodes; j++) {
            edge.vertices[j] = tensor->inds[j].data[i];
        }
        hypergraph->edges[i] = edge;
    }
    return hypergraph;
}

void free_hypergraph(Hypergraph* hypergraph) {
    for (sptNnzIndex i = 0; i < hypergraph->num_edges; i++) {
        free(hypergraph->edges[i].vertices);
    }
    free(hypergraph->edges);
    free(hypergraph);
}

/**
 * Initialize a MaxHeap.
 * @param capacity capacity of the heap
 * @return pointer to the created heap
 */
MaxHeap* create_max_heap(sptNnzIndex capacity) {
    MaxHeap* heap = (MaxHeap*)malloc(sizeof(MaxHeap));
    heap->capacity = capacity;
    heap->size = 0;
    heap->nodes = (MaxHeapNode*)malloc(capacity * sizeof(MaxHeapNode));
    return heap;
}

void free_max_heap(MaxHeap* heap) {
    free(heap->nodes);
    free(heap);
}

/**
 * Swap nodes in a MaxHeap.
 * @param node1
 * @param node2
 */
void swap_max_heap_nodes(MaxHeapNode* node1, MaxHeapNode* node2) {
    MaxHeapNode temp = *node1;
    *node1 = *node2;
    *node2 = temp;
}

/**
 * Max heapify a MaxHeap.
 * @param heap heap
 * @param index index of array of nodes to check from
 * @param in_heap [x] gives location of x in heap nodes array
 */
void max_heap_max_heapify(MaxHeap* heap, sptNnzIndex index, sptNnzIndex* in_heap) {
    sptNnzIndex largest = index;
    sptNnzIndex left = 2 * index + 1;
    sptNnzIndex right = 2 * index + 2;

    if (left < heap->size && compare_max_heap_nodes(&heap->nodes[left], &heap->nodes[largest]) > 0) {
        largest = left;
    }

    if (right < heap->size && compare_max_heap_nodes(&heap->nodes[right], &heap->nodes[largest]) > 0) {
        largest = right;
    }

    if (largest != index) {
        in_heap[heap->nodes[index].index] = largest + 1;
        in_heap[heap->nodes[largest].index] = index + 1;
        swap_max_heap_nodes(&heap->nodes[index], &heap->nodes[largest]);
        max_heap_max_heapify(heap, largest, in_heap);
    }
}

/**
 * Insert a node into the MaxHeap.
 * @param heap heap
 * @param index node index (NOT index in heap)
 * @param prim_key primary key/weight
 * @param sec_key secondary key/weight
 * @param in_heap [x] gives location of x in heap nodes array
 */
void insert_max_heap_node(MaxHeap* heap, sptNnzIndex index, sptNnzIndex prim_key, sptNnzIndex sec_key, sptNnzIndex* in_heap) {
    if (heap->size == heap->capacity) {
        printf("\nError: Heap overflow\n");
        return;
    }

    MaxHeapNode newNode;
    newNode.index = index;
    newNode.prim_key = prim_key;
    newNode.sec_key = sec_key;
    heap->nodes[heap->size] = newNode;
    sptNnzIndex i = heap->size;
    heap->size++;

    while (i != 0 && compare_max_heap_nodes(&heap->nodes[(i - 1) / 2], &heap->nodes[i]) < 0) {
        in_heap[heap->nodes[(i - 1) / 2].index] = i + 1;
        in_heap[heap->nodes[i].index] = ((i - 1) / 2) + 1;
        swap_max_heap_nodes(&heap->nodes[(i - 1) / 2], &heap->nodes[i]);
        i = (i - 1) / 2;
    }
}

/**
 * Extract (and remove) the max node in a MaxHeap.
 * @param heap heap
 * @param in_heap [x] gives location of x in heap nodes array
 * @return max node (removed)
 */
MaxHeapNode max_heap_extract_max(MaxHeap* heap, sptNnzIndex* in_heap) {
    if (heap->size == 0) {
        printf("\nERROR: Heap Underflow.\n");
        exit(1);
    }

    MaxHeapNode maxNode = heap->nodes[0];
    heap->nodes[0] = heap->nodes[heap->size - 1];
    heap->size--;

    in_heap[maxNode.index] = 0;
    in_heap[heap->nodes[0].index] = 1;

    max_heap_max_heapify(heap, 0, in_heap);

    return maxNode;
}

/**
 * Update a node in a max_heap.
 * Due to specific implementation reasons, only the primary key/weight can be updated.
 * @param heap heap
 * @param index node index (NOT index in heap)
 * @param prim_key new primary key/weight
 * @param in_heap [x] gives location of x in heap nodes array
 */
void update_max_heap_node(MaxHeap* heap, sptNnzIndex index, sptNnzIndex prim_key, sptNnzIndex* in_heap) {
    sptNnzIndex i = in_heap[index] - 1;
    heap->nodes[i].prim_key = prim_key;

    while (i != 0 && compare_max_heap_nodes(&heap->nodes[(i - 1) / 2], &heap->nodes[i]) < 0) {

        in_heap[heap->nodes[(i - 1) / 2].index] = i + 1;
        in_heap[heap->nodes[i].index] = ((i - 1) / 2) + 1;
        swap_max_heap_nodes(&heap->nodes[(i - 1) / 2], &heap->nodes[i]);

        i = (i - 1) / 2;
    }
}

/**
 * Run bfs_mcs reordering for a single node.
 * This thing takes a bunch of variables because a lot of info is precomputed to save time and memory.
 * Breaks from the paper's implementation in this way.
 * @param tensor tensor
 * @param mode mode
 * @param new_indices array to hold new indices
 * @param hypergraph hypergraph
 * @param edges_in_mode_in_vertex precomputed
 * @param num_edges_in_mode_in_vertex precomputed
 */
void spt_bfs_mcs_order_for_mode(
        sptSparseTensor* tensor,
        sptNnzIndex mode,
        sptIndex * new_indices,
        Hypergraph* hypergraph,
        sptNnzIndex*** edges_in_mode_in_vertex,
        sptNnzIndex** num_edges_in_mode_in_vertex
) {

    // printf("\n\t %d: BEGIN ALLOCATION 0 \n", mode);

    sptNnzIndex* prim_keys = (sptNnzIndex*) calloc(tensor->ndims[mode], sizeof(sptNnzIndex));
    sptNnzIndex* sec_keys = (sptNnzIndex*) calloc(tensor->ndims[mode], sizeof(sptNnzIndex));
    /** This holds the heap index of the node plus 1, OR 0 if not in heap */
    sptNnzIndex* in_heap = (sptNnzIndex*) calloc(tensor->ndims[mode], sizeof(sptNnzIndex));
    /**
     * Pointers to lists of marked indices for each mode.
     * [0][1] gives whether vertex 1 of mode 0 is marked (1 if yes).
     */
    sptNnzIndex** marked_in_vertex_in_mode = malloc(tensor->nmodes * sizeof(sptNnzIndex*));
    for (sptNnzIndex i = 0; i < tensor->nmodes; i++) {
        marked_in_vertex_in_mode[i] = calloc(tensor->ndims[i], sizeof(sptNnzIndex));
    }

    // New way of computing secondary keys, deviates from paper.
    // Had to do this for efficiency reasons.
    for (sptNnzIndex j = 0; j < hypergraph->num_edges; j++) {
        Hyperedge edge = hypergraph->edges[j];
        sec_keys[edge.vertices[mode]]++;
    }

    // printf("\n\t %d: BEGIN ALLOCATION 1 \n", mode);

    MaxHeap* heap = create_max_heap(tensor->ndims[mode]);
    for (sptNnzIndex i = 0; i < tensor->ndims[mode]; i++) {

        // This is the old way of computing the secondary key.
        // Moved it out of this loop for efficiency.
        // Deviates from the algorithm in the paper but same result.
        //
        // sptIndex degree2 = 0;
        // for (sptNnzIndex j = 0; j < hypergraph.num_edges; j++) {
        //     Hyperedge edge = hypergraph.edges[j];
        //     if (edge.vertices[mode] == i) {
        //         degree2++;
        //     }
        // }
        // insert_max_heap_node(heap, i, prim_keys[i], degree2);

        insert_max_heap_node(heap, i, prim_keys[i], sec_keys[i], in_heap);
    }

    // printf("\n\t %d: BEGIN MAIN \n", mode);

    // Everything below this point should look (somewhat) like the algorithm in the paper

    for (sptNnzIndex i = 0; i < tensor->ndims[mode]; i++) {

        // printf("\n\t\t %d \n", i);

        MaxHeapNode max_node = max_heap_extract_max(heap, in_heap);
        new_indices[max_node.index] = i;
        marked_in_vertex_in_mode[mode][max_node.index] = 1;

        // Prior to efficiency improvements
        // for (sptNnzIndex e = 0; e < hypergraph.num_edges; e++) {

        for (sptNnzIndex ei = 0; ei < num_edges_in_mode_in_vertex[mode][max_node.index]; ei++) {

            sptNnzIndex e = edges_in_mode_in_vertex[mode][max_node.index][ei];

            Hyperedge edge = hypergraph->edges[e];

            if (edge.vertices[mode] != max_node.index) {
                continue;
            }

            for (sptNnzIndex v = 0; v < edge.num_vertices; v++) {

                sptNnzIndex vertex = edge.vertices[v];

                if (v == mode || marked_in_vertex_in_mode[v][vertex] == 1) {
                    continue;
                }

                marked_in_vertex_in_mode[v][vertex] = 1;

                // Prior to efficiency improvements
                // for (sptNnzIndex e2 = 0; e2 < hypergraph.num_edges; e2++) {

                for (sptNnzIndex ej = 0; ej < num_edges_in_mode_in_vertex[v][vertex]; ej++) {

                    sptNnzIndex e2 = edges_in_mode_in_vertex[v][vertex][ej];
                    Hyperedge edge2 = hypergraph->edges[e2];

                    if (edge2.vertices[v] != vertex || e == e2) {
                        continue;
                    }

                    sptNnzIndex vertex2 = edge2.vertices[mode];

                    if (in_heap[vertex2] == 0) {
                        // not in heap
                        continue;
                    }

                    prim_keys[vertex2]++;
                    update_max_heap_node(heap, vertex2, prim_keys[vertex2], in_heap);
                }
            }
        }
    }

    // printf("\n\t %d: BEGIN FREE \n", mode);

    // Free everything
    free_max_heap(heap);
    free(prim_keys);
    free(sec_keys);
    free(in_heap);
    for (sptNnzIndex i = 0; i < tensor->nmodes; i++) {
        free(marked_in_vertex_in_mode[i]);
    }
    free(marked_in_vertex_in_mode);

    // printf("\n\t %d: DONE \n", mode);
}

/**
 * Run bfc_mcs_order reordering on a given sparse tensor.
 * @param tensor sparse tensor
 * @param new_indices new indices array to store reordering info.
 *
 * Note: new_indices will end up being structured as follows:
 *
 * newIndices[0][0...n_0-1] gives the new ids for dim 0,
 * newIndices[1][0...n_1-1] gives the new ids for dim 1,
 * newIndices[d-1][0...n_{d-1}-1] gives the new ids for dim d-1.
 */
void spt_bfs_mcs_order(sptSparseTensor* tensor, sptIndex** new_indices) {

    Hypergraph* hypergraph = create_hypergraph_from_spt_sparse_tensor(tensor);

    /**
     * IDs of edges in a mode associated with a vertex.
     * [0][1] is a pointer to the list of edge ids that associate with vertex 1 of mode 0.
     * I am precomputing this instead of computing on the fly for efficiency purposes.
     */
    sptNnzIndex*** edges_in_mode_in_vertex = malloc(tensor->nmodes * sizeof(sptNnzIndex**));

    /**
     * Length of the lists references in edges_in_mode_in_vertex.
     * [0][1] gives the length of the list at edges_in_mode_in_vertex[0][1].
     * I am precomputing this instead of computing on the fly for efficiency purposes.
     */
    sptNnzIndex** num_edges_in_mode_in_vertex = malloc(tensor->nmodes * sizeof(sptNnzIndex*));

    // printf("\n BEGIN ALLOCATION 0 \n");

    // Allocate the above
    for (sptNnzIndex i = 0; i < tensor->nmodes; i++) {
        num_edges_in_mode_in_vertex[i] = calloc(tensor->ndims[i], sizeof(sptNnzIndex));
        edges_in_mode_in_vertex[i] = malloc(tensor->ndims[i] * sizeof(sptNnzIndex*));
    }

    // printf("\n BEGIN ALLOCATION 1 \n");

    // Allocate/fill the above
#pragma omp parallel for default(none) shared(tensor, hypergraph, num_edges_in_mode_in_vertex, edges_in_mode_in_vertex)
    for (sptNnzIndex j = 0; j < tensor->nmodes; j++) {
        for (sptNnzIndex i = 0; i < hypergraph->num_edges; i++) {
            Hyperedge edge = hypergraph->edges[i];
            sptNnzIndex vert = edge.vertices[j];
            num_edges_in_mode_in_vertex[j][vert]++;
            // These lists are going to vary in size and I do not want to make them ALL the max size due to memory.
            // As such, I reallocate the main list on the fly when adding new elements.
            // I am sure there is a better way to do this, although I need to look into that later.
            if (num_edges_in_mode_in_vertex[j][vert] == 1) {
                edges_in_mode_in_vertex[j][vert] = malloc(1 * sizeof(sptNnzIndex));
            } else {
                edges_in_mode_in_vertex[j][vert] = realloc(edges_in_mode_in_vertex[j][vert], num_edges_in_mode_in_vertex[j][vert] * sizeof(sptNnzIndex));
            }
            edges_in_mode_in_vertex[j][vert][num_edges_in_mode_in_vertex[j][vert] - 1] = i;
        }
    }

    // printf("\n BEGIN MAIN \n");

#pragma omp parallel for default(none) shared(tensor, new_indices, hypergraph, edges_in_mode_in_vertex, num_edges_in_mode_in_vertex)
    for (sptNnzIndex i = 0; i < tensor->nmodes; i++) {
        spt_bfs_mcs_order_for_mode(tensor, i, new_indices[i], hypergraph, edges_in_mode_in_vertex, num_edges_in_mode_in_vertex);
    }

    // printf("\n BEGIN FREE \n");

    free_hypergraph(hypergraph);
    for (sptNnzIndex i = 0; i < tensor->nmodes; i++) {
        for (sptNnzIndex j = 0; j < tensor->ndims[i]; j++) {
            if (num_edges_in_mode_in_vertex[i][j] == 0) continue; // nothing allocated if true
            free(edges_in_mode_in_vertex[i][j]);
        }
        free(edges_in_mode_in_vertex[i]);
    }
    free(edges_in_mode_in_vertex);
    for (sptNnzIndex i = 0; i < tensor->nmodes; i++) {
        free(num_edges_in_mode_in_vertex[i]);
    }
    free(num_edges_in_mode_in_vertex);

    // printf("\n DONE \n");

}

/********************** MY BFS_MCS_ORDER IMPLEMENTATION END *************************/

/********************** MY LEXI_ORDER IMPLEMENTATION START *************************/

/**
 * Comparison function for quicksort for spt_lexi_order_quick_sort.
 * Compares two entries in a sparse tensor.
 * @param tensor sparse tensor
 * @param mode_order modes and order that we take into account for quicksort
 * @param a index of entry a
 * @param b index of entry b
 */
int spt_lexi_order_compare(sptSparseTensor* tensor, sptIndex* mode_order, sptNnzIndex a, sptNnzIndex b) {
    for (sptIndex i = 0; i < tensor->nmodes - 1; i++) {
        if (tensor->inds[mode_order[i]].data[a] < tensor->inds[mode_order[i]].data[b]) return -1;
        if (tensor->inds[mode_order[i]].data[a] > tensor->inds[mode_order[i]].data[b]) return 1;
    }
    return 0;
}

/**
 * Swap two entries in a sparse tensor.
 * @param tensor sparse tensor
 * @param mode_order modes and order that we take into account for quicksort
 * @param a index of entry a
 * @param b index of entry b
 */
void spt_swap_entries(sptSparseTensor* tensor, sptNnzIndex a, sptNnzIndex b) {
    for (sptIndex i = 0; i < tensor->nmodes; i++) {
        sptIndex temp = tensor->inds[i].data[a];
        tensor->inds[i].data[a] = tensor->inds[i].data[b];
        tensor->inds[i].data[b] = temp;
    }
    // Swap values
    sptValue temp_val = tensor->values.data[a];
    tensor->values.data[a] = tensor->values.data[b];
    tensor->values.data[b] = temp_val;
}

/**
 * Quicksort used by lexi_order_* algorithm.
 * @param tensor sparse tensor
 * @param mode_order modes and order that we take into account for quicksort
 * @param left index of left-most entry
 * @param right index of right most entry plus 1
 */
void spt_lexi_order_quick_sort(sptSparseTensor* tensor, sptIndex* mode_order, sptNnzIndex left, sptNnzIndex right) {

    /**
     * I refactored this from my original implementation to look more like the ParTI implementation.
     * My original implementation followed this source:
     * https://sites.radford.edu/~cshing/310/Lecture/PDC/parallelQuicksort.pdf
     * However, due to the data types and the specifics of how this library works, that was unsuccessful.
     *
     * Numbers and indices are in the sptNnzIndex and sptIndex formats, so perhaps them being uints is causing issues?
     * I imagine that that is why the original implementation is doing things like checking differences, rather than just greater/less than.
     */

    if (right - left < 2) {
        return;
    }

    /**
     * Modified implementation of Hoare partition scheme:
     * https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
     */
    sptNnzIndex pivot = (left + right) / 2;
    sptNnzIndex i, j;
    for (i = left, j = right - 1; ; ++i, --j) {
        while (spt_lexi_order_compare(tensor, mode_order, i, pivot) < 0) {
            i++;
        }
        while (spt_lexi_order_compare(tensor, mode_order, pivot, j) < 0) {
            j--;
        }
        if (i >= j) {
            break;
        }
        spt_swap_entries(tensor, i, j);
        if (i == pivot) {
            pivot = j;
        } else if (j == pivot) {
            pivot = i;
        }
    }

    /**
     * Originally I had both of the below go to new threads, but I commented one out.
     * It makes more sense to have the parent do one of the calculations.
     * Otherwise the parent calls 2 children and then doesn't do anything (see taskwait below).
     */
#pragma omp task default(none) firstprivate(left,i) shared(tensor,mode_order)
    spt_lexi_order_quick_sort(tensor, mode_order, left, i);
    // #pragma omp task firstprivate(i,right) shared(mode_order,tensor)
    spt_lexi_order_quick_sort(tensor, mode_order, i, right);

    // bring it all together
#pragma omp taskwait
}

/**
 * Performs lexi_order re-ordering for a given mode.
 * @param tensor sparse tensor
 * @param mode mode
 * @param og_indices arr[0...n_0-1] WILL give the original indices for given mode
 */
void spt_lexi_order_for_mode(sptSparseTensor* tensor, sptIndex mode, sptIndex* og_indices) {

    /**
     * This basically just stores what modes we actually care about for quicksort.
     * This is needed because we want to sort all non-zeros along all BUT the given mode.
     * We can then pass this to helper functions, so they know what modes to care about.
     */
    sptIndex * mode_order = malloc((tensor->nmodes - 1) * sizeof(sptIndex));
    sptIndex m = 0;
    for (sptIndex i = 0; i < tensor->nmodes - 1; i++) {
        if (m == mode) m++;
        mode_order[i] = m;
        m++;
    }

    // start sort timer
    double t0 = u_seconds();

    /**
     * This is our parallel section.
     * The helper function has OMP tasks to run in parallel.
     * We have the nowait because there is an implicit barrier from parallel already.
     */
#pragma omp parallel default(none) shared(tensor,mode_order)
#pragma omp single nowait
    spt_lexi_order_quick_sort(tensor, mode_order, 0, tensor->nnz);

    // record sort timer
    double t1 = u_seconds()-t0;
    printf("mode %u, sort time %.2f\n", mode, t1);

    /**
     * ParTI's implementation for this code had all matrix arrays and indices start from 1.
     * I am not entirely sure why, I assume that it is something specific to do with the library.
     * In any case, I have integrated some of their code for that to meet that standard.
     */

    // begin matricizing the tensor
    sptNnzIndex* row_ptrs = malloc((tensor->nnz + 2) * sizeof(sptNnzIndex));
    sptIndex* col_idxs = malloc((tensor->nnz + 2) * sizeof(sptIndex));
    sptNnzIndex num_rows = 1;
    sptNnzIndex curr_row = 1;
    // start create timer
    t0 = u_seconds();
    // add initial values to first positions
    row_ptrs[0] = 0;
    row_ptrs[1] = 1;
    col_idxs[1] = tensor->inds[mode].data[0] + 1;
    // do check-loop from second entry onwards
    for (sptNnzIndex i = 2; i < tensor->nnz + 1; i++) {
        if (spt_lexi_order_compare(tensor, mode_order, i - 2, i - 1) != 0) {
            // if we need to move to a new row
            curr_row++;
            num_rows++;
            row_ptrs[curr_row] = i;
        }
        // add mode's idx to cols
        col_idxs[i] = tensor->inds[mode].data[i - 1] + 1;
    }
    // unsure why we need this but it breaks otherwise
    row_ptrs[curr_row + 1] = tensor->nnz + 1;
    // we over-allocated space for row_ptrs and now we fix that
    row_ptrs = realloc(row_ptrs, (num_rows + 2) * sizeof(sptNnzIndex));
    // record create timer
    t1 =u_seconds()-t0;
    printf("mode %u, create time %.2f\n", mode, t1);

    // this will hold the current permutation
    sptIndex* c_perm = malloc((tensor->ndims[mode] + 1) * sizeof(sptIndex));

    // start lex_order timer
    t0 = u_seconds();

    /**
     * I am not re-implementing this function, since it was not the main focus of the paper.
     * This is essentially a variation of orderly refine.
     */
    lexOrderThem(num_rows, tensor->ndims[mode], row_ptrs, col_idxs, c_perm);

    // record lex_order timer
    t1 =u_seconds()-t0;
    printf("mode %u, lexorder time %.2f\n", mode, t1);

    // "normal" c_perm (zero-indexed, revert increments)
    sptIndex* c_perm_normal = malloc((tensor->ndims[mode] + 1) * sizeof(sptIndex));
    for (sptIndex i = 0; i < tensor->ndims[mode]; i++) {
        c_perm_normal[i] = c_perm[i + 1] - 1;
    }

    // inverted "normal" (zero-indexed, revert increments) c_perm
    sptIndex* c_perm_normal_invert = malloc((tensor->ndims[mode] + 1) * sizeof(sptIndex));
    for (sptIndex i = 0; i < tensor->ndims[mode]; i++) {
        c_perm_normal_invert[c_perm_normal[i]] = i;
    }

    // store the old values of og_indices, as we will modify it soon
    sptIndex* og_indices_old = malloc((tensor->ndims[mode] + 1) * sizeof(sptIndex));
    for (sptIndex i = 0; i < tensor->ndims[mode]; i++) {
        og_indices_old[i] = og_indices[i];
    }

    // update og_indices
    for (sptIndex i = 0; i < tensor->ndims[mode]; i ++) {
        og_indices[i] = og_indices_old[c_perm_normal[i]];
    }

    // update tensor indices
    for (sptNnzIndex i = 0; i < tensor->nnz; i++) {
        tensor->inds[mode].data[i] = c_perm_normal_invert[tensor->inds[mode].data[i]];
    }

    free(og_indices_old);
    free(c_perm_normal_invert);
    free(c_perm_normal);
    free(c_perm);
    free(row_ptrs);
    free(col_idxs);
    free(mode_order);
}

/**
 * Run lexi_order reordering on the given sparse tensor.
 * @param tsr sparse tensor
 * @param newIndices array allocated such that [m][i] holds new index of i in mode m
 * @param iterations number of iterations to run the algorithm
 */
void spt_lexi_order(sptSparseTensor* tsr, sptIndex** newIndices, sptIndex iterations){

    sptIndex** og_indices = malloc(tsr->nmodes * sizeof(sptIndex*));
    for (sptIndex i = 0; i < tsr->nmodes; i++) {
        og_indices[i] = malloc(tsr->ndims[i] * sizeof(sptIndex));
    }
    for (sptIndex i = 0; i < tsr->nmodes; i++) {
        for (sptIndex j = 0; j < tsr->ndims[i]; j++) {
            og_indices[i][j] = (sptIndex) j;
        }
    }

    for (sptIndex i = 0; i < iterations; i++) {
        printf("[Lexi-order] Optimizing the numbering for iteration %u\n", i + 1);
        for (sptIndex j = 0; j < tsr->nmodes; j++) {
            spt_lexi_order_for_mode(tsr, (sptIndex) j, og_indices[j]);
        }
    }

    for (sptIndex i = 0; i < tsr->nmodes; i++) {
        for (sptIndex j = 0; j < tsr->ndims[i]; j++) {
            // TODO: explain why we are doing this
            newIndices[i][og_indices[i][j]] = j;
        }
    }

    for (sptIndex i = 0; i < tsr->nmodes; i++) {
        free(og_indices[i]);
    }
    free(og_indices);
}

/********************** MY LEXI_ORDER IMPLEMENTATION END *************************/