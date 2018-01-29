#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <stack>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
using namespace std;

enum DATA { UNDISCOVERED, DISCOVERED, INTERNAL, SURFACE, EXTERNAL};

struct Projection
{
	Eigen::Matrix<float, 3, 4> m_matProjection;
	cv::Mat m_matImage;
	const uint m_threshold = 125;

	bool IsOutofRange(int x, int max)
	{
		return x < 0 || x >= max;
	}

	bool IsInModel(double x, double y, double z)
	{
		Eigen::Vector3f vec3 = m_matProjection * Eigen::Vector4f(x, y, z, 1);
		int indX = vec3[1] / vec3[2];
		int indY = vec3[0] / vec3[2];

		if (IsOutofRange(indX, m_matImage.size().height) || IsOutofRange(indY, m_matImage.size().width)) return false;
		else return m_matImage.at<uchar>((uint)(indX), (uint)(indY)) > m_threshold;
	}
};

struct ProjectionList
{
	vector<Projection> m_vecProjectionList;

	void LoadMatrix(const string & strFileName);
	void LoadImage(const string &strDir, const string &strPrefix, const string &strSuffix);
	bool CheckInModel(double x, double y, double z);
};

void ProjectionList::LoadMatrix(const string & strFileName)
{
	ifstream fin(strFileName);
	int nDummy;
	Eigen::Matrix<float, 3, 3> matInt;
	Eigen::Matrix<float, 3, 4> matExt;
	while (fin >> nDummy)
	{
		Projection temp;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				fin >> matInt(i, j);
		double dDummy;
		fin >> dDummy >> dDummy;

		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 4; ++j)
				fin >> matExt(i, j);

		temp.m_matProjection = matInt * matExt;

		m_vecProjectionList.emplace_back(temp);
	}
}

void ProjectionList::LoadImage(const string &strDir, const string &strPrefix, const string &strSuffix)
{
	string strFileName(strDir);
	strFileName += '/';
	strFileName += strPrefix;
	for (uint i = 0; i < m_vecProjectionList.size(); ++i)
	{
		m_vecProjectionList[i].m_matImage = cv::imread(strFileName + to_string(i) + strSuffix, CV_8UC1);
	}
}

bool ProjectionList::CheckInModel(double x, double y, double z)
{
	bool bResult = true;
	for (auto &prj : m_vecProjectionList)
	{
		if (!prj.IsInModel(x, y, z))
		{
			bResult = false;
			break;
		}
	}
	return bResult;
}


struct ScaleTransform
{
	int m_resolution;
	double m_min;
	double m_max;
	double index2coor(int index)
	{
		return m_min + index * (m_max - m_min) / m_resolution;
	}

	ScaleTransform(int resolution, double min, double max)
		: m_resolution(resolution), m_min(min), m_max(max)
	{}
};

class Model
{
	using ModelCore = vector<vector<vector<char>>>;
	using Point = Eigen::Vector3i;

	typedef vector<vector<bool>> Pixel;
	typedef vector<Pixel> Voxel;

public:
	Model(int resX, int resY, int resZ, int nNN, int nRelax);
	void SaveWithoutNormal(const string & strFileName);
	void SaveWithNormal(const string & strFileName);
	void SetupProjections(const string & strParam, const string &strDir, const string &strPrefix, const string &strSuffix);
	void ComputeModelDFS();

private:
	ScaleTransform m_scaleX;
	ScaleTransform m_scaleY;
	ScaleTransform m_scaleZ;
	ProjectionList m_projectionList;
	ModelCore m_coreData;
	int m_nNN;
	int m_nRelax;
	Eigen::Vector3f ComputeNormal(int indX, int indY, int indZ);
};

Model::Model(int resX, int resY, int resZ, int nNN, int nRelax)
	: m_scaleX(resX, -5, 5)
	, m_scaleY(resY, -10, 10)
	, m_scaleZ(resZ, 15, 30)
{
	m_nNN = nNN;
	m_nRelax = nRelax;
	m_coreData = vector<vector<vector<char>>>(m_scaleX.m_resolution, vector<vector<char>>(m_scaleY.m_resolution, vector<char>(m_scaleZ.m_resolution, UNDISCOVERED)));
}


void Model::SaveWithoutNormal(const string & strFileName)
{
	ofstream fout(strFileName);
	
	for (int i = 0; i < m_scaleX.m_resolution; ++i)
	{
		for (int j = 0; j < m_scaleY.m_resolution; ++j)
		{
			for (int k = 0; k < m_scaleZ.m_resolution; ++k)
			{
				if (m_coreData[i][j][k] == SURFACE)
				{
					double nCx = m_scaleX.index2coor(i);
					double nCy = m_scaleY.index2coor(j);
					double nCz = m_scaleZ.index2coor(k);
					fout << nCx << ' ' << nCy << ' ' << nCz << endl;
				}
			}
		}
	}
}

void Model::SaveWithNormal(const string & strFileName)
{
	ofstream fout(strFileName);

	for (int i = 0; i < m_scaleX.m_resolution; ++i)
	{
		for (int j = 0; j < m_scaleY.m_resolution; ++j)
		{
			for (int k = 0; k < m_scaleZ.m_resolution; ++k)
			{
				if (m_coreData[i][j][k] == SURFACE)
				{
					double nCx = m_scaleX.index2coor(i);
					double nCy = m_scaleY.index2coor(j);
					double nCz = m_scaleZ.index2coor(k);

					fout << nCx << ' ' << nCy << ' ' << nCz << ' ';
					Eigen::Vector3f nor = ComputeNormal(i, j, k);
					fout << nor[0] << ' ' << nor[1] << ' ' << nor[2] << std::endl;
				}
			}
		}
	}
}


void Model::SetupProjections(const string & strParam, const string & strDir, const string & strPrefix, const string & strSuffix)
{
	m_projectionList.LoadMatrix(strParam);
	m_projectionList.LoadImage(strDir, strPrefix, strSuffix);
}

void Model::ComputeModelDFS()
{
	int xInit = 0.5 * m_scaleX.m_resolution;
	int yInit = 0.5 * m_scaleY.m_resolution;
	int zInit = 0.5 * m_scaleZ.m_resolution;

	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };

	/*int DX[8] = { 1,1,-1,-1,1,1,-1,-1 };
	int DY[8] = { -1,1,1,-1,-1,1,-1,-1 };
	int DZ[8] = { 1,1,1,1,-1,-1,-1,-1 };*/

	Point ptInit(xInit, yInit, zInit);
	stack<Point> stkDFS;
	stkDFS.push(ptInit);
	m_coreData[ptInit[0]][ptInit[1]][ptInit[2]] = DISCOVERED;

	while (!stkDFS.empty())
	{
		Point ptSearch(stkDFS.top());
		stkDFS.pop();

		if (m_coreData[ptSearch[0]][ptSearch[1]][ptSearch[2]] != DISCOVERED) continue;

		int nStatus = INTERNAL;

		for (int i = 0; i < 6; ++i)
		{
			Point ptNeighbour(ptSearch[0] + dx[i], ptSearch[1] + dy[i], ptSearch[2] + dz[i]);

			if (m_coreData[ptNeighbour[0]][ptNeighbour[1]][ptNeighbour[2]] == INTERNAL ||
				m_coreData[ptNeighbour[0]][ptNeighbour[1]][ptNeighbour[2]] == DISCOVERED ||
				m_coreData[ptNeighbour[0]][ptNeighbour[1]][ptNeighbour[2]] == SURFACE) continue;

			if (m_coreData[ptNeighbour[0]][ptNeighbour[1]][ptNeighbour[2]] == EXTERNAL)
			{
				nStatus = SURFACE;
				break;
			}

			double nNCx = m_scaleX.index2coor(ptNeighbour[0]);
			double nNCy = m_scaleY.index2coor(ptNeighbour[1]);
			double nNCz = m_scaleZ.index2coor(ptNeighbour[2]);
			
			if (m_projectionList.CheckInModel(nNCx, nNCy, nNCz))
			{
				if (m_coreData[ptNeighbour[0]][ptNeighbour[1]][ptNeighbour[2]] == UNDISCOVERED)
				{
					m_coreData[ptNeighbour[0]][ptNeighbour[1]][ptNeighbour[2]] = DISCOVERED;
					stkDFS.push(ptNeighbour);
				}
			}
			else
			{
				nStatus = SURFACE;
				m_coreData[ptNeighbour[0]][ptNeighbour[1]][ptNeighbour[2]] = EXTERNAL;
				break;
			}
		}

		m_coreData[ptSearch[0]][ptSearch[1]][ptSearch[2]] = nStatus;
		

		//Some useless hack
		/*if (m_coreData[ptSearch[0]][ptSearch[1]][ptSearch[2]] == INTERNAL && m_nRelax > 0)
		{
			bool bDeep = true;
			for (int i = 0; i < 8; ++i)
			{
				int x = ptSearch[0] + m_nRelax * DX[i];
				int y = ptSearch[1] + m_nRelax * DY[i];
				int z = ptSearch[2] + m_nRelax * DZ[i];

				if (m_coreData[x][y][z] != INTERNAL && m_coreData[x][y][z] != DISCOVERED)
				{
					bDeep = false;
					break;
				}
			}

			if (bDeep)
			{
				for (int dx = -m_nRelax; dx <= m_nRelax; ++dx)
				{
					for (int dy = -m_nRelax; dy <= m_nRelax; ++dy)
					{
						for (int dz = -m_nRelax; dz <= m_nRelax; ++dz)
						{
							m_coreData[ptSearch[0] + dx][ptSearch[1] + dy][ptSearch[2] + dz] = INTERNAL;
						}
					}
				}
			}
		}*/
	}
	
}

Eigen::Vector3f Model::ComputeNormal(int indX, int indY, int indZ)
{
	auto IsOutofRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_scaleX.m_resolution
			|| indexY >= m_scaleY.m_resolution
			|| indexZ >= m_scaleZ.m_resolution;
	};

	vector<Eigen::Vector3f> vecSurface;
	vector<Eigen::Vector3f> vecInner;

	for (int dX = -m_nNN; dX <= m_nNN; dX++)
		for (int dY = -m_nNN; dY <= m_nNN; dY++)
			for (int dZ = -m_nNN; dZ <= m_nNN; dZ++)
			{
				if ((dX == 0) && (dY == 0) && (dZ == 0))
				{
					continue;
				}

				Point ptNeighbour(indX + dX, indY + dY, indZ + dZ);

				if (!IsOutofRange(ptNeighbour[0], ptNeighbour[1], ptNeighbour[2]))
				{
					float coorX = m_scaleX.index2coor(ptNeighbour[0]);
					float coorY = m_scaleY.index2coor(ptNeighbour[1]);
					float coorZ = m_scaleZ.index2coor(ptNeighbour[2]);
					if (m_coreData[ptNeighbour[0]][ptNeighbour[1]][ptNeighbour[2]]== SURFACE)
						vecSurface.emplace_back(Eigen::Vector3f(coorX, coorY, coorZ));
					else if (m_coreData[ptNeighbour[0]][ptNeighbour[1]][ptNeighbour[2]] == INTERNAL)
						vecInner.emplace_back(Eigen::Vector3f(coorX, coorY, coorZ));
				}
			}

	Eigen::Vector3f point(m_scaleX.index2coor(indX), m_scaleY.index2coor(indY), m_scaleZ.index2coor(indZ));

	Eigen::MatrixXf matA(3, vecSurface.size());
	for (int i = 0; i < vecSurface.size(); i++)
		matA.col(i) = vecSurface[i] - point;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);

	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	for (auto const& vec : vecInner)
		innerCenter += vec;
	innerCenter /= vecInner.size();

	if (normalVector.dot(point - innerCenter) < 0)
		normalVector *= -1;
	return normalVector;
}

int main()
{
	using namespace std;
	using namespace chrono;

	auto chrT0 = system_clock::now();

	Model model(300, 300, 300, 3, 0);
	model.SetupProjections("../../calibParamsI.txt", "../../wd_segmented", "WD2_", "_00020_segmented.png");

	auto chrT1 = system_clock::now();
	cout << "准备数据用时：" << duration_cast<milliseconds>(chrT1 - chrT0).count() << "ms" << endl;

	model.ComputeModelDFS();
	auto chrT2 = system_clock::now();
	cout << "生成模型用时：" << duration_cast<milliseconds>(chrT2 - chrT1).count() << "ms" << endl;

	model.SaveWithoutNormal("../../WithoutNormal.xyz");
	auto chrT3 = system_clock::now();
	cout << "Non-Normal用时：" << duration_cast<milliseconds>(chrT3 - chrT2).count() << "ms" << endl;

	model.SaveWithNormal("../../WithNormal.xyz");
	auto chrT4 = system_clock::now();
	cout << "Normal用时：" << duration_cast<milliseconds>(chrT4 - chrT3).count() << "ms" << endl;

	system("PoissonRecon.x64 --in ../../WithNormal.xyz --out ../../mesh.ply");
	auto chrT5 = system_clock::now();
	cout << "Poisson重建用时：" << duration_cast<milliseconds>(chrT5 - chrT4).count() << "ms" << endl;

	cout << "整体用时：" << duration_cast<milliseconds>(chrT5 - chrT0).count() << "ms" << endl;

	return 0;
}