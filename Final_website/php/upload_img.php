<?php

include "config.php";
session_start();
$user= $_SESSION["loggedin_user"];

$sql="SELECT * from patients WHERE email_id ='".$_SESSION["loggedin_user"]."'";
$result=$conn->query($sql);
while ($row=mysqli_fetch_assoc($result))
{
    $_SESSION["id"]=$row['id'];
    $_SESSION["name"] = $row['name'];
    $_SESSION["age"] = $row['age'];
    $_SESSION["dob"] = $row['date of birth'];
    $_SESSION["sex"] = $row['sex'];
    $_SESSION["blood_group"] = $row['blood_group'];
    $_SESSION["contact"] = $row['contact'];
    $_SESSION["address"] = $row['address'];
    $_SESSION["medical_history"] = $row['medical_history'];
    $_SESSION["email"] = $row['email_id'];
    
}
$file = $_FILES['file'];


if($file['size'] != 0)
{
    $file_name = $file['name'];
    $file_temp = $file['tmp_name'];
    $file_size = $file['size'];
    $file_error = $file['error'];


    $file_ext = explode('.',$file_name);
    $file_ext = strtolower(end($file_ext));

    $allowed = array ('tif','tiff','jpg','jpeg','png' );

    if (in_array($file_ext,$allowed))
    {
        if($file_error == 0)
        {
            if($file_size <= 10485760)
            {   
                $directoryName = 'uploads/' . $user;
                if(!is_dir($directoryName))
                {
                    //Directory does not exist, so create it.
                    mkdir($directoryName, 0777, true);
                }

                $file_name_new = uniqid('',true) . '.' . $file_ext;
                $file_destination = $directoryName. "/" . $file_name_new;

                if(move_uploaded_file($file_temp,$file_destination))
                {
                    #echo $file_destination;
                    #sending details to python file
                    #dont forget chmod 777 txt file otherwise it wont open 
                    $myfile = fopen("mytxt.txt", "w") or die("Unable to open file!");
                    fwrite($myfile, ($file_destination. "+". $user));
                    fclose($myfile);

                    ####insert into users table
                    $sql = " UPDATE `patients` SET `cell_image` = '$file_destination' WHERE `email_id` = '$user' " ;
                    $conn->query($sql);
                    
                    #chmod 777 to python file as well
                    
                    #$command = escapeshellcmd('/var/www/html/php/myScript.py');
                    #$output = shell_exec($command);
                    $output = shell_exec('python /var/www/html/php/myScript.py');

                    
                    
                    
                    #echo $output;
                    
                    
                }
            }


        }
    }

}
else
{
            echo ("<script>alert('You better upload an image file and it should be less than 10 mb !');window.location.href='../upload_img.php';</script>" ) ;
            exit(); #break the entire flow 
}


$myfile = fopen("myreport.txt", "r") or die("Unable to open file!");
$final_result = fgets($myfile);


fclose($myfile);


########################################################################################################################
#                                                  profile code                                                        #
########################################################################################################################


$date_of_report = date("Y-m-d");
$time_of_report = date("h:i:sa");


$sql = "INSERT INTO `report` (`email`, `path`, `date`, `time`, `result`) VALUES ('$user', '$file_destination', '$date_of_report', '$time_of_report', '$final_result')";


if ($conn->query($sql) == TRUE)
	{
        
	}     		
	else 
	{
    	$sql= "UPDATE `report` SET  `path` = '$file_destination', `date` = '$date_of_report', `time` = '$time_of_report' WHERE `report`.`email` = '$user'";
        $result = $conn->query($sql);
        if ($result==1)
        {
            echo "<script> alert('cool!!! Saved your report successfully!!!'); </script> ";
        }
        else
        {
            
            echo "<script> alert('Error'); </script> ";
        }
        
    }



$conn->close();

?>








<!DOCTYPE html>
<html>

<head>

    <meta charset="utf-8">
    <title>Report</title>
    <link rel="stylesheet" href="../css/report.css" media="all" />



</head>

<body class="body1">
     
        <header class="header1">
            <div class="clearfix">
                <div id="logo">
                    <img src="../images/144315.svg">
                </div>
                <h1>Medical Report</h1>
                <div id="company" class="clearfix">
                    <div>APSIT HEALTH CENTER</div>
                    <div>Ghodbunder Rd, opp. Hypercity Mall,<br /> Kasarvadavali, Thane West, Thane,</div>
                    <div>Maharashtra 400615</div>
                    <div>+91-123456789</div>
                    <div><a href="mailto:company@example.com">apsitmlproject@gmail.com</a></div>
                </div>
                <div id="project">
                    <div><span>Patitent Name </span>   <?php echo $_SESSION["name"] ; ?> </div>
                    <div><span>Address <Address</span> <?php echo $_SESSION["address"] ; ?> </div>
                    <div><span>DATE Of Birth </span>   <?php echo $_SESSION["dob"] ; ?></div>
                    
                    <div><span>Report Date </span>     <?php echo $date_of_report;?></div>
                    
                    <div><span>Report Time </span>     <?php echo $time_of_report;?></div>
                </div>
            </div>
    </header>
    <hr>
    <h3>Histopathologic Cancer Report For Patitent : <?php echo $_SESSION["id"] ; ?></h3>
    <div> Clinical Diagnosis : <?php echo $final_result; ?> </div>

    <div> Date of Diagnosis : <?php echo $date_of_report;?> </div>

    <div><h4> Gross Description: <h4> </div>
    <div><p>The specimen is received in two parts. They are labeled #1, "biopsy bladder tumor", and #2, "scalene node, left". 
        Part #1 consists of multiple fragments of gray-brown tissue which appear slightly hemorrhagic. 
        They are submitted in their entirety for processing. Part #2 consists of multiple fragments of fatty yellow tissue 
        which range in size from 0.2 to 1.0 cm in diameter. 
        They are submitted in their entirety for processing.</p></div>
    
        
    <div><h4>Microscopic:  <h4> </div>
    <div> <p>Section of bladder contains areas of transitional cell carcinoma. 
    No area of invasion can be identified. A marked acute and chronic inflammatory 
    reaction with eosinophils is noted together with some necrosis. Sections are examined at six levels. 
    Section of lymph node contains normal node with reactive germinal centers.All sections taken radially 
    from the superficial center of the resection site fail to include tumor, indicating the tumor to have 
    originated deep within the breast parenchyma. Similarly, there is no malignancy in the nipple region, 
    or in the lactiferous sinuses.Sections of deep surgical margin demonstrate diffuse tumor infiltration
    of deep fatty tissues, however, there is no invasion of muscle. Total size of primary tumor is estimated 
    to be 4 cm in greatest dimension.</p> </div>

    <div>
    </div>
  
    <div class="footer">
    
    Report was created on a computer and is valid without the signature and seal.
    <br>
        <button onclick="printinvoice()">Print Report</button>
    <a href="../index.php">Home</a>
    </div>
    <script>
            function printinvoice()
            {
                window.print();
            }

    </script>
    
    </div>
</body>
</html>