/* Modify these parameters carefully */

#define TRAINING_DATA	"iris.data"
#define TESTING_DATA	"iris.data.check"
#define NO_TRAINING_DATA	150
#define NO_TESTING_DATA	18
#define FEATURES	4
#define LABELS	3
#define NO_OF_LAYERS	1
#define ETA	0.1
#define EPOCHS	1000

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

float **create_mat(int n,int m);
void print_mat(float **arr,int n,int m);
float **input_data(FILE *fp,float ***y,int n,int m,int no_of_label);
void error(float ***w,int N,int d,int L,int *NL,int label);
int main()
{
	int count,i,j,d=FEATURES,N=NO_TRAINING_DATA,ep,n,L=NO_OF_LAYERS,k,h,label=LABELS;
	/*	N - no. of inputs, d -dimensions or attributes excluding bias,
		L no of hidden layers 1<=l<=L(output layer)
		h for layer iteration
		input comes from ith (l-1)th layer and move into the jth output layer
	*/
	int NL[L+1];
	/*
		number of nodes excluding the bias in all lth layers 0<=l<=L
		NL[0] for atrributes in the input layer excluding bias
	*/
	NL[0]=d;	//nodes in input layer fixed depends on attributes
	NL[L]=label;	//nodes in O/P layer is fixed depends on number of labels

	/* Modify these parameters as per your requirement*/
	NL[1]=3;
	//NL[2]=3;


	float sum=0,***w,lr=ETA,**del,**input,**y,**s,**x;
	/*
		w for weights each layer has 2-D weight matrix
		w[0] points to 1st hidden layer weight matrix
		lr - learning rate
		s - signal i.e w(transpose)*x
		x theta(s) i.e sigmoid(s)
		each hidden layer has theta s ie. 1-D matrix
		x[1] points to 1st hidden layer theta(s) 1-D matrix
		x[0] simply points the input layer
	*/
	w=(float ***)malloc(L*sizeof(float **));
	s=(float **)malloc((L+1)*sizeof(float *));	//signal
	x=(float **)malloc((L+1)*sizeof(float *));	//theta_x
	del=(float **)malloc(L*sizeof(float *));	//del


	/*
		w is 3-D array w[l][j][i] it means lth layer jth output and ith input
		del is 2-D array del[l][j] it means lth layer and jth output delta value
	*/

	/* no of w is no of layers */
	for(i=0; i<L; i++)
	{
		/*
			creating matrix w[i] say 1st layer no of node in 1st layer * no. of attrs+1
			creating matrix del[i] say 1st layer  it will have all deltas on first layer
		*/
		w[i]=create_mat(NL[i+1],NL[i]+1);
		del[i]=(float *)malloc((NL[i+1]+1)*sizeof(float));
	}
	/* no of x increases by one because of x[0] is input layer  */
	for(i=0; i<L+1; i++)
	{
		s[i]=(float *)malloc((NL[i]+1)*sizeof(float));
		x[i]=(float *)malloc((NL[i]+1)*sizeof(float));
		s[i][0]=1;
		x[i][0]=1;	//bias
	}
	srand(time(NULL));
	for(k=0; k<L; k++)
	{
		for(j=0; j<NL[k+1]; j++)
		{
			for(i=0; i<NL[k]+1; i++)
			{
				/* for checking it is used */
				//w[k][j][i]=(j+1)/10.0;
				/* randomly initialised between 0 to .1 */
				w[k][j][i]=(rand()%100)/100.0;
			}
		}
	}

	printf("Before Training\n");
	for(i=0; i<L; i++)
	{
		printf("\tW%d\n",i+1);
		print_mat(w[i],NL[i+1],NL[i]+1);
	}

	FILE *fp;
	fp=fopen(TRAINING_DATA,"r");
	/* y for output labels d+1 attributes 1 is for */
	input=input_data(fp,&y,N,d+1,label);	//input is (N x d+1) 2-D array
	fclose(fp);
	//print_mat(input,N,d+1);
	//print_mat(y,N,label);
	for(ep=0; ep<EPOCHS; ep++)
	{
		/*	Stochastic Gradient Descent	*/
		//n=rand()%N;
		n=0;
		count=0;
		//printf("%d\n",n);
		while(count!=N)
		{
			/* taking the input (say first) input to x[0]	*/
			for(i=0; i<d+1; i++)
			{
				x[0][i]=input[n][i];
			}
			/* Computing signal i.e w_t*x and theta_x=sigmoid(signal)	*/
			for(h=1; h<=L; h++)
			{
				for(j=0; j<NL[h]; j++)
				{
					s[h][j+1]=0;
					for(i=0; i<NL[h-1]+1; i++)
					{
						s[h][j+1]+=w[h-1][j][i]*x[h-1][i];
					}
					x[h][j+1]=1.0/(1+exp(-s[h][j+1]));
				}
			}

			//printf("%f %f\n",x[h-1][1],x[h-1][2] );

			/* calculating delta for the output layer only */
			for(j=0; j<NL[L]; j++)
			{
				del[L-1][j]=(y[n][j]-x[L][j+1])*x[L][j+1]*(1-x[L][j+1]);
			}

			/* calculating deltas for all the hidden layers */
			for(h=L; h>1; h--)
			{
				for(j=0; j<NL[h-1]; j++)
				{
					sum=0;
					for(k=0; k<NL[h]; k++)
					{
						sum+=del[h-1][k]*w[h-1][k][j];
					}
					del[h-2][j]=x[h-1][j+1]*(1-x[h-1][j+1])*sum;
				}
			}

			/* Updating output and hidden layer weights */
			for(h=L; h>0; h--)
			{
				for(j=0; j<NL[h]; j++)
				{
					for(i=0; i<NL[h-1]+1; i++)
					{
						w[h-1][j][i]+=lr*del[h-1][j]*x[h-1][i];
					}
				}
			}
			count++;
			n=(n+1)%N;
		}

	}
	printf("After Training\n");
	for(i=0; i<L; i++)
	{
		printf("\tW%d\n",i+1);
		print_mat(w[i],NL[i+1],NL[i]+1);
	}

	error(w,NO_TESTING_DATA,d,L,NL,label);
	return 0;
}

void error(float ***w,int N,int d,int L,int *NL,int label)
{
	int n,j,i,h,maxindex,correct=0;
	float **input,**y,**s,**x,E_out;
	FILE *fp;
	fp=fopen(TESTING_DATA,"r");
	input=input_data(fp,&y,N,d+1,label);
	fclose(fp);
	s=(float **)malloc((L+1)*sizeof(float *));	//signal
	x=(float **)malloc((L+1)*sizeof(float *));	//theta_x
	for(i=0; i<L+1; i++)
	{
		s[i]=(float *)malloc((NL[i]+1)*sizeof(float));
		x[i]=(float *)malloc((NL[i]+1)*sizeof(float));
		s[i][0]=1;
		x[i][0]=1;	//bias
	}
	printf("\tChecking the error\n");
	for(n=0; n<N; n++)
	{
		for(i=0; i<d+1; i++)
		{
			x[0][i]=input[n][i];
		}
		for(h=1; h<=L; h++)
		{
			for(j=0; j<NL[h]; j++)
			{
				s[h][j+1]=0;
				for(i=0; i<NL[h-1]+1; i++)
				{
					s[h][j+1]+=w[h-1][j][i]*x[h-1][i];
				}
				x[h][j+1]=1.0/(1+exp(-s[h][j+1]));
			}
		}
		maxindex=1;
		for(i=1; i<NL[L]; i++)
		{
			if(x[L][i+1]>x[L][maxindex])
				maxindex=i+1;
		}
		if(y[n][maxindex-1]==1)
		{
			correct++;
		}
	}
	/* Percentage is wild card use %% to print it */
	printf("Accuracy = %.3f %%\n",(correct*100.0)/N);
}

float **input_data(FILE *fp,float ***y,int n,int m,int no_of_label)
{
	float **arr,**z;
	int i=0,j;
	arr=create_mat(n,m);
	z=create_mat(n,no_of_label);
	/* Avoid use of while(!feof(fp)) might causes error in checking */
	while(i<n)
	{
		arr[i][0]=1;
		for(j=1; j<m; j++)
			fscanf(fp,"%f,",&arr[i][j]);
		for(j=0; j<no_of_label; j++)
			fscanf(fp,"%f,",&z[i][j]);
		i++;
	}
	*y=z;
	return arr;
}

float **create_mat(int n, int m)
{
	int i;
	float **arr;
	arr=(float **)calloc(n,sizeof(float *));
	for(i=0; i<n; i++)
		arr[i]=(float *)calloc(m,sizeof(float));
	return arr;
}

void print_mat(float **arr,int n, int m)
{
	int i,j;
	/* for better print in the terminal */
	printf("+");
	for(i=0; i<8*m; i++)
		printf("-");
	printf("+\n");
	for(i=0; i<n; i++)
	{
		printf("|");
		for(j=0; j<m; j++)
		{
			printf("%.3f\t",arr[i][j]);
		}
		printf(" |\n");
	}
	/* for better print in the terminal */
	printf("+");
	for(i=0; i<8*m; i++)
		printf("-");
	printf("+\n");
}
