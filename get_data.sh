# Creates a Data repository

mkdir Data
cd Data



# Creates the sub-repositories and downloads the files in it

# Dubrovnik

mkdir Dubrovnik
cd Dubrovnik

prefix=https://grail.cs.washington.edu/projects/bal/data/dubrovnik/

declare -a L=(problem-16-22106-pre.txt.bz2 problem-88-64298-pre.txt.bz2 problem-135-90642-pre.txt.bz2 problem-142-93602-pre.txt.bz2 problem-150-95821-pre.txt.bz2 problem-161-103832-pre.txt.bz2 problem-173-111908-pre.txt.bz2 problem-182-116770-pre.txt.bz2 problem-202-132796-pre.txt.bz2 problem-237-154414-pre.txt.bz2 problem-253-163691-pre.txt.bz2 problem-262-169354-pre.txt.bz2 problem-273-176305-pre.txt.bz2 problem-287-182023-pre.txt.bz2 problem-308-195089-pre.txt.bz2 problem-356-226730-pre.txt.bz2)

for file in ${L[@]}
do
  wget "$prefix$file"
done


# Final

cd ..
mkdir Final
cd Final

prefix=https://grail.cs.washington.edu/projects/bal/data/final/

declare -a L=(problem-93-61203-pre.txt.bz2 problem-394-100368-pre.txt.bz2 problem-871-527480-pre.txt.bz2 problem-961-187103-pre.txt.bz2 problem-1936-649673-pre.txt.bz2 problem-3068-310854-pre.txt.bz2 problem-4585-1324582-pre.txt.bz2 problem-13682-4456117-pre.txt.bz2)

for file in ${L[@]}
do
  wget "$prefix$file"
done

# LadyBug

cd ..
mkdir LadyBug
cd LadyBug

prefix=https://grail.cs.washington.edu/projects/bal/data/ladybug/

declare -a L=("problem-49-7776-pre.txt.bz2" "problem-73-11032-pre.txt.bz2" "problem-138-19878-pre.txt.bz2" "problem-318-41628-pre.txt.bz2" "problem-372-47423-pre.txt.bz2" "problem-412-52215-pre.txt.bz2" "problem-460-56811-pre.txt.bz2" "problem-539-65220-pre.txt.bz2" "problem-598-69218-pre.txt.bz2" "problem-646-73584-pre.txt.bz2" "problem-707-78455-pre.txt.bz2" "problem-783-84444-pre.txt.bz2" "problem-810-88814-pre.txt.bz2" "problem-856-93344-pre.txt.bz2" "problem-885-97473-pre.txt.bz2" "problem-931-102699-pre.txt.bz2" "problem-969-105826-pre.txt.bz2" "problem-1031-110968-pre.txt.bz2" "problem-1064-113655-pre.txt.bz2" "problem-1118-118384-pre.txt.bz2" "problem-1152-122269-pre.txt.bz2" "problem-1197-126327-pre.txt.bz2" "problem-1235-129634-pre.txt.bz2" "problem-1266-132593-pre.txt.bz2" "problem-1340-137079-pre.txt.bz2" "problem-1469-145199-pre.txt.bz2" "problem-1514-147317-pre.txt.bz2" "problem-1587-150845-pre.txt.bz2" "problem-1642-153820-pre.txt.bz2" "problem-1695-155710-pre.txt.bz2" "problem-1723-156502-pre.txt.bz2")

for file in ${L[@]}
do
  wget "$prefix$file"
done

# Trafalgar

cd ..
mkdir Trafalgar
cd Trafalgar

prefix=https://grail.cs.washington.edu/projects/bal/data/trafalgar/

declare -a L=(problem-21-11315-pre.txt.bz2 problem-39-18060-pre.txt.bz2 problem-50-20431-pre.txt.bz2 problem-126-40037-pre.txt.bz2 problem-138-44033-pre.txt.bz2 problem-161-48126-pre.txt.bz2 problem-170-49267-pre.txt.bz2 problem-174-50489-pre.txt.bz2 problem-193-53101-pre.txt.bz2 problem-201-54427-pre.txt.bz2 problem-206-54562-pre.txt.bz2 problem-215-55910-pre.txt.bz2 problem-225-57665-pre.txt.bz2 problem-257-65132-pre.txt.bz2)

for file in ${L[@]}
do
  wget "$prefix$file"
done

# LadyBug

cd ..
mkdir Venice
cd Venice

prefix=https://grail.cs.washington.edu/projects/bal/data/venice/

declare -a L=(problem-52-64053-pre.txt.bz2 problem-89-110973-pre.txt.bz2 problem-245-198739-pre.txt.bz2 problem-427-310384-pre.txt.bz2 problem-744-543562-pre.txt.bz2 problem-951-708276-pre.txt.bz2 problem-1102-780462-pre.txt.bz2 problem-1158-802917-pre.txt.bz2 problem-1184-816583-pre.txt.bz2 problem-1238-843534-pre.txt.bz2 problem-1288-866452-pre.txt.bz2 problem-1350-894716-pre.txt.bz2 problem-1408-912229-pre.txt.bz2 problem-1425-916895-pre.txt.bz2 problem-1473-930345-pre.txt.bz2 problem-1490-935273-pre.txt.bz2 problem-1521-939551-pre.txt.bz2 problem-1544-942409-pre.txt.bz2 problem-1638-976803-pre.txt.bz2 problem-1666-983911-pre.txt.bz2 problem-1672-986962-pre.txt.bz2 problem-1681-983415-pre.txt.bz2 problem-1682-983268-pre.txt.bz2 problem-1684-983269-pre.txt.bz2 problem-1695-984689-pre.txt.bz2 problem-1696-984816-pre.txt.bz2 problem-1706-985529-pre.txt.bz2 problem-1776-993909-pre.txt.bz2 problem-1778-993923-pre.txt.bz2)

for file in ${L[@]}
do
  wget "$prefix$file"
done

cd ../..
